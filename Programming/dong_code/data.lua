require "utils/SymbolsManager.lua"

function process_train_data(opt)
  require('pl.stringx').import()
  require 'pl.seq'

  local timer = torch.Timer()
  
  local data = {}

  local word_manager = SymbolsManager(true)
  word_manager:init_from_file(path.join(opt.data_dir, 'vocab.txt'), opt)

  local user_manager = SymbolsManager(true)
  local product_manager = SymbolsManager(true)
  local rating_manager = SymbolsManager(true)

  print('loading text file...')
  local f = torch.DiskFile(path.join(opt.data_dir, opt.train .. '.txt'), 'r', true)
  f:clearError()
  local rawdata = f:readString('*l')
  while (not f:hasError()) do
    local l_list = rawdata:strip():split('\t')
    local u = user_manager:add_symbol(tonumber(l_list[1]))
    local p = product_manager:add_symbol(tonumber(l_list[2]))
    local r = rating_manager:add_symbol(l_list[3])
    local w_list = word_manager:get_symbol_idx_for_list(l_list[4]:split(' '))
    table.insert(data,{u,p,r,w_list})
    -- read next line
    rawdata = f:readString('*l')
  end
  f:close()

  print(string.format('#data: %d\t#vocab: %d\t#user: %d\t#product: %d\t#rating: %d', #data,
    word_manager.vocab_size - 3, user_manager.vocab_size - 3, product_manager.vocab_size - 3, rating_manager.vocab_size - 3))

  collectgarbage()

  -- save output preprocessed files
  local out_mapfile = path.join(opt.data_dir, 'map.t7')
  print('saving ' .. out_mapfile)
  torch.save(out_mapfile, {word_manager, user_manager, product_manager, rating_manager})

  collectgarbage()

  local out_datafile = path.join(opt.data_dir, opt.train .. '.t7')
  print('saving ' .. out_datafile)
  torch.save(out_datafile, data)

  collectgarbage()
end

function serialize_data(opt, name)
  require('pl.stringx').import()
  require 'pl.seq'

  local fn = path.join(opt.data_dir, name .. '.txt')

  if not path.exists(fn) then
    print('no file: ' .. fn)
    return nil
  end

  local timer = torch.Timer()
  
  local word_manager, user_manager, product_manager, rating_manager = unpack(torch.load(path.join(opt.data_dir, 'map.t7')))

  local data = {}

  print('loading text file...')
  local f = torch.DiskFile(fn, 'r', true)
  f:clearError()
  local rawdata = f:readString('*l')
  while (not f:hasError()) do
    local l_list = rawdata:strip():split('\t')
    local u = user_manager:get_symbol_idx(tonumber(l_list[1]))
    local p = product_manager:get_symbol_idx(tonumber(l_list[2]))
    local r = rating_manager:get_symbol_idx(l_list[3])
    local w_list = word_manager:get_symbol_idx_for_list(l_list[4]:split(' '))
    table.insert(data,{u,p,r,w_list})
    -- read next line
    rawdata = f:readString('*l')
  end
  f:close()

  print(string.format('#data: %d', #data))

  collectgarbage()

  -- save output preprocessed files
  local out_datafile = path.join(opt.data_dir, name .. '.t7')

  print('saving ' .. out_datafile)
  torch.save(out_datafile, data)
end

local cmd = torch.CmdLine()
cmd:option('-data_dir', '/disk/scratch_ssd/lidong/gen_review/books/', 'data directory')
cmd:option('-train', 'train', 'train data path')
cmd:option('-dev', 'dev', 'dev data path')
cmd:option('-test', 'test', 'test data path')
cmd:option('-min_freq', 15, 'minimum word frequency')
cmd:option('-max_vocab_size', 15000, 'maximum vocabulary size')
cmd:text()
opt = cmd:parse(arg)

process_train_data(opt)
serialize_data(opt, opt.dev)
serialize_data(opt, opt.test)
