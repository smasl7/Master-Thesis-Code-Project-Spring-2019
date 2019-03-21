require 'torch'
require 'nn'
require 'nngraph'
-- nngraph.setDebug(true)
require 'optim'
require 'lfs'
require 'utils/SymbolsManager'
include "../utils/utils.lua"
include "../utils/bleu.lua"
local MinibatchLoader = require 'utils.MinibatchLoader'
-- include "layers/Embedding.lua"
-- include "layers/MaskedClassNLLCriterion.lua"

function transfer_data(x)
  if opt.gpuid >= 0 then
    return x:cuda()
  end
  return x
end

function convert_to_string(idx_list, f_out)
  local w_list = {}
  for i = 1, #idx_list do
    table.insert(w_list, word_manager:get_idx_symbol(idx_list[i]))
  end
  return table.concat(w_list, ' ')
end

cmd = torch.CmdLine()
cmd:text()
cmd:text('Sample from the learned model')
cmd:text()
cmd:text('Options')
cmd:option('-model', 'model checkpoint to use for sampling')
cmd:option('-data_dir', '/disk/scratch_ssd/lidong/gen_review/books/', 'data directory')
cmd:option('-input', 'test.t7', 'input data filename')
cmd:option('-seed', 123,'random number generator\'s seed')
cmd:option('-sample', 0,' 0 to use max at each timestep (-beam_size = 1), 1 to sample at each timestep, 2 to beam search')
cmd:option('-beam_size', 20,'beam size')
cmd:option('-batch_size', 100,'batch size')
cmd:option('-gpuid',0, 'which gpu to use. -1 = use CPU')
cmd:option('-display', 1,'whether display on console')
cmd:text()

-- parse input params
opt = cmd:parse(arg)
opt.output = opt.model .. '.sample'

-- initialize gpu/cpu
init_device(opt)

-- load the model checkpoint
if not lfs.attributes(opt.model, 'mode') then
  print('Error: File ' .. opt.model .. ' does not exist. Are you sure you didn\'t forget to prepend cv/ ?')
end
checkpoint = torch.load(opt.model)
lookup = checkpoint.lookup
encoder = checkpoint.encoder
dec_rnn_unit = checkpoint.rnn_unit
dec_att_unit = checkpoint.dec_att_unit

lookup:evaluate()
encoder:evaluate()
dec_rnn_unit:evaluate()
dec_att_unit:evaluate()

-- initialize the rnn state to all zeros
s = {}
for i = 1, checkpoint.opt.num_layers do
  -- c and h for all layers
  table.insert(s, transfer_data(torch.zeros(opt.batch_size, checkpoint.opt.rnn_size)))
  table.insert(s, transfer_data(torch.zeros(opt.batch_size, checkpoint.opt.rnn_size)))
end
lookup_result = {}
for d = 1, 3 do
  lookup_result[d] = transfer_data(torch.zeros(opt.batch_size, checkpoint.opt.upr_size))
end
m_generate = transfer_data(torch.zeros(opt.batch_size, checkpoint.opt.seq_length))

-- initialize the vocabulary manager to display text
word_manager, user_manager, product_manager, rating_manager = unpack(torch.load(path.join(opt.data_dir, 'map.t7')))
-- load data
-- local data = torch.load(path.join(opt.data_dir, opt.input))
local test_loader = MinibatchLoader.create(opt, 'test')

-- require('mobdebug').start('129.215.91.104')

local f_out = torch.DiskFile(opt.output, 'w')
local reference_list = {}
local candidate_list = {}
local batch_list = test_loader:all_batch()
for i_batch = 1, #batch_list do
  -- load batch data
  local attribute_batch, text_batch = unpack(batch_list[i_batch])
  -- ship batch data to gpu
  if opt.gpuid >= 0 then
    attribute_batch[1] = attribute_batch[1]:float():cuda()
    attribute_batch[2] = attribute_batch[2]:float():cuda()
    attribute_batch[3] = attribute_batch[3]:float():cuda()
    text_batch = text_batch:float():cuda()
  end

  -- encode
  copy_table(lookup_result, lookup:forward(attribute_batch))
  copy_table(s, encoder:forward(lookup_result))

  local prev_word
  if opt.gpuid >= 0 then
    prev_word = (torch.Tensor(opt.batch_size):fill(word_manager:get_symbol_idx('<S>'))):float():cuda()
  else
    prev_word = torch.Tensor(opt.batch_size):fill(word_manager:get_symbol_idx('<S>'))
  end

  local len = 0
  local not_end_table = {}
  for i = 1, opt.batch_size do
    table.insert(not_end_table, i)
  end
  while true do
    len = len + 1

    -- forward the rnn for next word
    local s_cur = dec_rnn_unit:forward({prev_word, s})
    copy_table(s, s_cur)
    local prediction = dec_att_unit:forward({lookup_result, s[2 * checkpoint.opt.num_layers]})
    
    -- log probabilities from the previous timestep
    local _, _prev_word = prediction:max(2)
    prev_word = _prev_word:resize(opt.batch_size)
    m_generate[{{}, len}]:copy(prev_word)

    -- find which has been finished
    local cur_not_end_set = {}
    for _, v in ipairs(not_end_table) do
      if (prev_word[v] ~= word_manager:get_symbol_idx('<E>')) then
        table.insert(cur_not_end_set, v)
      end
    end
    not_end_table = cur_not_end_set

    if (#not_end_table == 0) or (len >= checkpoint.opt.seq_length) then
      break
    end
  end

  for i = 1, opt.batch_size do
    local candidate ={}
    local cand_view = m_generate[{i,{}}]
    for j = 1, cand_view:nElement() do
      if (cand_view[j] ~= word_manager:get_symbol_idx('<E>')) then
        table.insert(candidate, cand_view[j])
      else
        break
      end
    end
    local reference = {}
    local ref_view = text_batch[{i,{}}]
    -- skip <S>
    for j = 2, ref_view:nElement() do
      if (ref_view[j] ~= word_manager:get_symbol_idx('<E>')) then
        table.insert(reference, ref_view[j])
      else
        break
      end
    end

    table.insert(reference_list, reference)
    table.insert(candidate_list, candidate)

    local ref_str = convert_to_string(reference)
    local cand_str = convert_to_string(candidate)
    -- print to console
    if opt.display > 0 then
      print(ref_str)
      print(cand_str)
      print(' ')
    end
    -- write to file
    f_out:writeString(ref_str)
    f_out:writeString('\n')
    f_out:writeString(cand_str)
    f_out:writeString('\n')
  end
  
  if i_batch % 50 == 0 then
    collectgarbage()
  end
end

-- compute evaluation metric
local val_bleu = compute_bleu(candidate_list, reference_list, 4)
print('BLEU = ' .. val_bleu)
f_out:writeString('BLEU = ' .. val_bleu)
f_out:writeString('\n')
val_bleu = compute_bleu(candidate_list, reference_list, 1)
print('BLEU = ' .. val_bleu)
f_out:writeString('BLEU = ' .. val_bleu)

f_out:close()
