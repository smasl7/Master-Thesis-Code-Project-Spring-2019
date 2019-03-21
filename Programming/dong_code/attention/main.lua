require 'torch'
require 'nn'
require 'nngraph'
-- nngraph.setDebug(true)
require 'optim'
require('pl.stringx').import()
require 'pl.seq'
require 'utils/SymbolsManager'
-- include "../layers/Embedding.lua"
include "../utils/utils.lua"
local MinibatchLoader = require 'utils.MinibatchLoader'
-- include "layers/ClassNLLCriterion.lua"
-- ProFi = require 'ProFi'

function transfer_data(x)
  if opt.gpuid>=0 then
    return x:cuda()
  end
  return x
end

function lstm(x, prev_c, prev_h)
  -- Calculate all four gates in one go
  local i2h = nn.Linear(opt.rnn_size, 4 * opt.rnn_size)(x)
  local h2h = nn.Linear(opt.rnn_size, 4 * opt.rnn_size)(prev_h)
  local gates = nn.CAddTable()({i2h, h2h})
  
  -- Reshape to (batch_size, n_gates, hid_size)
  -- Then slize the n_gates dimension
  local reshaped_gates =  nn.Reshape(4, opt.rnn_size)(gates)
  local sliced_gates = nn.SplitTable(2)(reshaped_gates)
  
  -- Use select gate to fetch each gate and apply nonlinearity
  local in_gate          = nn.Sigmoid()(nn.SelectTable(1)(sliced_gates))
  local in_transform     = nn.Tanh()(nn.SelectTable(2)(sliced_gates))
  local forget_gate      = nn.Sigmoid()(nn.SelectTable(3)(sliced_gates))
  local out_gate         = nn.Sigmoid()(nn.SelectTable(4)(sliced_gates))

  local next_c           = nn.CAddTable()({
      nn.CMulTable()({forget_gate, prev_c}),
      nn.CMulTable()({in_gate,     in_transform})
  })
  local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

  return next_c, next_h
end

function create_decoder_lstm_unit(w_size)
  -- input
  local x = nn.Identity()()
  local prev_s = nn.Identity()()

  local i = {[0] = nn.LookupTable(w_size, opt.rnn_size)(x)}
  local next_s = {}
  local splitted = {prev_s:split(2 * opt.num_layers)}
  for layer_idx = 1, opt.num_layers do
    local prev_c = splitted[2 * layer_idx - 1]
    local prev_h = splitted[2 * layer_idx]
    local x_in = i[layer_idx - 1]
    if opt.dropout > 0 then
      x_in = nn.Dropout(opt.dropout)(x_in)
    end
    local next_c, next_h = lstm(x_in, prev_c, prev_h)
    table.insert(next_s, next_c)
    table.insert(next_s, next_h)
    i[layer_idx] = next_h
  end
  local m = nn.gModule({x, prev_s}, {nn.Identity()(next_s)})
  
  return transfer_data(m)
end

function create_lookuptable(u_size, p_size, r_size)
  -- input
  local user = nn.Identity()()
  local product = nn.Identity()()
  local rating = nn.Identity()()
  -- lookup
  local u = nn.LookupTable(u_size, opt.upr_size)(user)
  local p = nn.LookupTable(p_size, opt.upr_size)(product)
  local r = nn.LookupTable(r_size, opt.upr_size)(rating)

  local upr_table = {u, p, r}

  local module = nn.gModule({user, product, rating}, {nn.Identity()(upr_table)})
  
  return transfer_data(module)
end

function create_encoder()
  -- input
  local upr_table = nn.Identity()()

  local last_layer = nn.JoinTable(2)({upr_table})
  if opt.dropout > 0 then
    last_layer = nn.Dropout(opt.dropout)(last_layer)
  end

  local last_dim = 3 * opt.upr_size
  
  -- hidden
  if #opt.encoder_hidden > 0 then
    for i = 1, #opt.encoder_hidden do
      local tmp = nn.Tanh()(nn.Linear(last_dim, opt.encoder_hidden[i])(last_layer))
      last_layer = tmp
      last_dim = opt.encoder_hidden[i]
    end
  end
  
  -- initialization of the first unit of LSTM
  local o = nn.Tanh()(nn.Linear(last_dim, opt.rnn_size * 2 * opt.num_layers)(last_layer))
  local reshaped_o = nn.Reshape(2 * opt.num_layers, opt.rnn_size)(o)
  local sliced_o = nn.SplitTable(2)(reshaped_o)

  local module = nn.gModule({upr_table}, {sliced_o})
  
  return transfer_data(module)
end

function create_attention_unit(w_size)
  -- input
  local upr_table = nn.Identity()()
  local dec_s_top = nn.Identity()()

  local score_table = {}
  local splitted = {upr_table:split(3)}
  for i = 1, 3 do
    table.insert(score_table, nn.Tanh()(nn.Linear(opt.upr_size + opt.rnn_size, 1)(nn.JoinTable(2)({splitted[i], dec_s_top}))))
  end
  -- (batch*3)
  local attention = nn.SoftMax()(nn.JoinTable(2)(score_table))
  -- (batch*3*H)^T * (batch*3*1) = (batch*H*1)
  local enc_attention = nn.MM(true, false)({nn.View(3, opt.upr_size):setNumInputDims(1)(nn.JoinTable(2)(upr_table)), nn.View(-1, 1):setNumInputDims(1)(attention)})
  local hid = nn.Tanh()(nn.Linear(opt.upr_size + opt.rnn_size, opt.rnn_size)(nn.JoinTable(2)({nn.Sum(3)(enc_attention), dec_s_top})))
  if opt.dropout > 0 then
    hid = nn.Dropout(opt.dropout)(hid)
  end
  local h2y = nn.Linear(opt.rnn_size, w_size)(hid)
  local pred = nn.LogSoftMax()(h2y)
  local m = nn.gModule({upr_table, dec_s_top}, {pred})
  
  return transfer_data(m)
end

function setup()
  -- initialize model
  model = {}
  model.s = {}
  model.ds = {}
  model.dlookup = {}
  
  for j = 0, opt.seq_length do
    model.s[j] = {}

    for d = 1, 2 * opt.num_layers do
      model.s[j][d] = transfer_data(torch.zeros(opt.batch_size, opt.rnn_size))
    end
  end

  for d = 1, 2 * opt.num_layers do
    model.ds[d] = transfer_data(torch.zeros(opt.batch_size, opt.rnn_size))
  end

  for d = 1, 3 do
    model.dlookup[d] = transfer_data(torch.zeros(opt.batch_size, opt.upr_size))
  end

  local word_manager, user_manager, product_manager, rating_manager = unpack(torch.load(path.join(opt.data_dir, 'map.t7')))

  print("Creating encoder")
  model.lookup = create_lookuptable(user_manager.vocab_size, product_manager.vocab_size, rating_manager.vocab_size)
  model.lookup:training()

  model.encoder = create_encoder()
  model.encoder:training()

  print("Creating decoder")
  model.rnn_unit = create_decoder_lstm_unit(word_manager.vocab_size)
  model.rnn_unit:training()

  model.dec_att_unit = create_attention_unit(word_manager.vocab_size)
  model.dec_att_unit:training()

  model.criterions = {}
  for i = 1, opt.seq_length do
    table.insert(model.criterions, transfer_data(nn.ClassNLLCriterion()))
  end

  -- collect all parameters to a vector
  param_x, param_dx = combine_all_parameters(model.lookup, model.encoder, model.rnn_unit, model.dec_att_unit)
  print('number of parameters in the model: ' .. param_x:nElement())
  
  param_x:uniform(-opt.init_weight, opt.init_weight)

  -- make a bunch of clones after flattening, as that reallocates memory (tips from char-rnn)
  model.rnns = cloneManyTimes(model.rnn_unit, opt.seq_length)
  model.dec_atts = cloneManyTimes(model.dec_att_unit, opt.seq_length)
end

function reset_ds()
  for d = 1, #model.ds do
    model.ds[d]:zero()
  end
end

function reset_dlookup()
  for d = 1, #model.dlookup do
    model.dlookup[d]:zero()
  end
end

function eval_training(param_x_)
  model.lookup:training()
  model.encoder:training()
  model.rnn_unit:training()
  model.dec_att_unit:training()
  for i = 1, #model.rnns do
    model.rnns[i]:training()
  end
  for i = 1, #model.dec_atts do
    model.dec_atts[i]:training()
  end

  -- load batch data
  local attribute_batch, text_batch = train_loader:random_batch()
  -- ship batch data to gpu
  if opt.gpuid >= 0 then
    attribute_batch[1] = attribute_batch[1]:float():cuda()
    attribute_batch[2] = attribute_batch[2]:float():cuda()
    attribute_batch[3] = attribute_batch[3]:float():cuda()
    text_batch = text_batch:float():cuda()
  end

  -- do not predict for <E>
  local max_len = text_batch:size(2) - 1

  -- forward propagation ===============================
  if param_x_ ~= param_x then
    param_x:copy(param_x_)
  end

  -- encode (initialize decoder using encoding results)
  local lookup_result = model.lookup:forward(attribute_batch)
  local encoding_result = model.encoder:forward(lookup_result)
  -- whether need to copy?
  copy_table(model.s[0], encoding_result)
  
  -- decode
  local softmax_predictions = {}
  local loss = 0
  for i = 1, max_len do
    model.s[i] = model.rnns[i]:forward({text_batch[{{}, i}], model.s[i - 1]})
    softmax_predictions[i] = model.dec_atts[i]:forward({lookup_result, model.s[i][2*opt.num_layers]})
    loss = loss + model.criterions[i]:forward(softmax_predictions[i], text_batch[{{}, i+1}])

    if opt.gpuid >= 0 then
      cutorch.synchronize()
    end
  end
  loss = loss / max_len

  -- backward propagation ===============================
  param_dx:zero()
  reset_ds()
  reset_dlookup()
  for i = max_len, 1, -1 do
    local crit_dx = model.criterions[i]:backward(softmax_predictions[i], text_batch[{{}, i+1}])
    local tmp1, tmp2 = unpack(model.dec_atts[i]:backward({lookup_result, model.s[i][2*opt.num_layers]}, crit_dx))
    add_table(model.dlookup, tmp1)
    model.ds[2*opt.num_layers]:add(tmp2)
    copy_table(model.ds, model.rnns[i]:backward({text_batch[{{}, i}], model.s[i - 1]}, model.ds)[2])
    if opt.gpuid >= 0 then
      cutorch.synchronize()
    end
  end
  -- back-propagate to encoder
  add_table(model.dlookup, model.encoder:backward(lookup_result, model.ds))
  model.lookup:backward(attribute_batch, model.dlookup)
  if opt.gpuid >= 0 then
    cutorch.synchronize()
  end

  -- clip gradient element-wise
  param_dx:clamp(-opt.grad_clip, opt.grad_clip)

  return loss, param_dx
end

function eval_split(data_split)
  model.lookup:evaluate()
  model.encoder:evaluate()
  model.rnn_unit:evaluate()
  model.dec_att_unit:evaluate()
  for i = 1, #model.rnns do
    model.rnns[i]:evaluate()
  end
  for i = 1, #model.dec_atts do
    model.dec_atts[i]:evaluate()
  end

  local loss = 0

  local batch_list = data_split:all_batch()
  for i = 1, #batch_list do
    -- load batch data
    local attribute_batch, text_batch = unpack(batch_list[i])
    -- ship batch data to gpu
    if opt.gpuid >= 0 then
      attribute_batch[1] = attribute_batch[1]:float():cuda()
      attribute_batch[2] = attribute_batch[2]:float():cuda()
      attribute_batch[3] = attribute_batch[3]:float():cuda()
      text_batch = text_batch:float():cuda()
    end

    -- do not predict for <E>
    local max_len = text_batch:size(2) - 1

    -- forward propagation ===============================
    -- encode (initialize decoder using encoding results)
    local lookup_result = model.lookup:forward(attribute_batch)
    local encoding_result = model.encoder:forward(lookup_result)
    -- whether need to copy?
    copy_table(model.s[0], encoding_result)
    
    -- decode
    local softmax_predictions = {}
    local loss_batch = 0
    for i = 1, max_len do
      model.s[i] = model.rnns[i]:forward({text_batch[{{}, i}], model.s[i - 1]})
      softmax_predictions[i] = model.dec_atts[i]:forward({lookup_result, model.s[i][2*opt.num_layers]})
      loss_batch = loss_batch + model.criterions[i]:forward(softmax_predictions[i], text_batch[{{}, i+1}])

      if opt.gpuid >= 0 then
        cutorch.synchronize()
      end
    end
    loss_batch = loss_batch / max_len

    loss = loss + loss_batch
  end

  loss = loss / #batch_list

  return loss
end

function main()
  local cmd = torch.CmdLine()
  cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
  cmd:option('-data_dir', '/disk/scratch_ssd/lidong/gen_review/books/', 'data path')
  -- bookkeeping
  cmd:option('-seed',123,'torch manual random number generator seed')
  cmd:option('-checkpoint_dir', '/disk/scratch_ssd/lidong/gen_review/cv', 'output directory where checkpoints get written')
  cmd:option('-savefile','save','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
  cmd:option('-print_every',500,'how many steps/minibatches between printing out the loss')
  cmd:option('-eval_val_every',10000,'every how many iterations should we evaluate on validation data?')
  cmd:option('-continue_model_path','','continue model path')
  -- model params
  cmd:option('-rnn_size', 512, 'size of LSTM internal state')
  cmd:option('-upr_size', 64, 'size of user/product/rating embedding')
  cmd:option('-encoder_hidden', '', 'hidden layer of encoder')
  cmd:option('-num_layers', 2, 'number of layers in the LSTM')
  -- optimization
  cmd:option('-dropout',0.2,'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
  cmd:option('-seq_length',70,'number of timesteps to unroll for')
  cmd:option('-batch_size',50,'number of sequences to train on in parallel')
  cmd:option('-max_epochs',100,'number of full passes through the training data')
  cmd:option('-learning_rate',0.002,'learning rate')
  cmd:option('-init_weight',0.08,'initailization weight')
  cmd:option('-learning_rate_decay',0.97,'learning rate decay')
  cmd:option('-learning_rate_decay_after',10,'in number of epochs, when to start decaying the learning rate')
  cmd:option('-decay_rate',0.95,'decay rate for rmsprop')
  cmd:option('-grad_clip',5,'clip gradients at this value')
  cmd:text()
  opt = cmd:parse(arg)

  if #opt.encoder_hidden > 0 then
    opt.encoder_hidden = seq(opt.encoder_hidden:split(',')):map(tonumber):copy()
  else
    opt.encoder_hidden = {}
  end

  -- initialize gpu/cpu
  init_device(opt)

  -- setup network
  setup()
  -- if continue train, load the pre-trained model
  if #opt.continue_model_path > 0 then
    print('continue train the model: ' .. opt.continue_model_path)
    local checkpoint = torch.load(opt.continue_model_path)
    model.encoder = checkpoint.encoder
    model.rnns = checkpoint.rnns
    model.rnn_unit = checkpoint.rnns[1]
    param_x, param_dx = combine_all_parameters(model.encoder, model.rnn_unit)
  end

  -- load data
  train_loader = MinibatchLoader.create(opt, 'train')
  local dev_loader = MinibatchLoader.create(opt, 'dev')

  -- make sure output directory exists
  if not path.exists(opt.checkpoint_dir) then
    lfs.mkdir(opt.checkpoint_dir)
  end

  -- start training
  local step = 0
  local epoch = 0
  local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
  
  -- require('mobdebug').start()
  
  -- ProFi:start()

  print("Starting training.")
  train_losses = {}
  val_losses = {}
  local iterations = opt.max_epochs * train_loader.num_batch
  local loss0 = nil
  local best_val_result = nil
  for i = 1, iterations do
    local epoch = i / train_loader.num_batch

    local timer = torch.Timer()
    local _, loss = optim.rmsprop(eval_training, param_x, optim_state)
    local time = timer:time().real

    local train_loss = loss[1] -- the loss is inside a list, pop it
    train_losses[i] = train_loss

    -- exponential learning rate decay
    if i % train_loader.num_batch == 0 and opt.learning_rate_decay < 1 then
      if epoch >= opt.learning_rate_decay_after then
        local decay_factor = opt.learning_rate_decay
        optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
        print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. optim_state.learningRate)
      end
    end

    if i % opt.print_every == 0 then
      local avg_train_loss = 0
      local k = 0
      for j = 1, opt.print_every do
        if i - j + 1 >= 1 then
          avg_train_loss = avg_train_loss + train_losses[i - j + 1]
          k = k + 1
        end
      end
      avg_train_loss = avg_train_loss / k
      print(string.format("%d/%d (epoch %.3f), avg_train_loss = %6.8f, train_loss = %6.8f, time/batch = %.2fs", i, iterations, epoch, avg_train_loss, train_loss, time))
    end

    -- every now and then or on last iteration
    if i % opt.eval_val_every == 0 or i == iterations then
      -- ProFi:stop()
      -- ProFi:writeReport('profile.txt')

      -- evaluate loss on validation data
      local val_loss = eval_split(dev_loader)
      -- print('val loss: ' .. val_loss)
      val_losses[i] = val_loss

      if best_val_result == nil or val_loss < best_val_result then
        local savefile = string.format('%s/%s_epoch%.2f_%.4f.t7', opt.checkpoint_dir, opt.savefile, epoch, val_loss)
        print('saving checkpoint to ' .. savefile)
        local checkpoint = {}
        checkpoint.lookup = model.lookup
        checkpoint.encoder = model.encoder
        checkpoint.rnn_unit = model.rnn_unit
        checkpoint.dec_att_unit = model.dec_att_unit
        checkpoint.opt = opt
        checkpoint.train_losses = train_losses
        checkpoint.val_loss = val_loss
        checkpoint.val_losses = val_losses
        checkpoint.i = i
        checkpoint.epoch = epoch

        torch.save(savefile, checkpoint)
      end
    end
   
    if i % 30 == 0 then
      collectgarbage()
    end

    -- handle early stopping if things are going really bad
    if loss[1] ~= loss[1] then
      print('loss is NaN.  This usually indicates a bug.')
      break -- halt
    end
    if loss0 == nil then
      loss0 = loss[1]
    end
    -- if loss[1] > loss0 * 3 then
    --   print('loss is exploding, aborting.')
    --   break -- halt
    -- end
  end
end

main()
