local MinibatchLoader = {}
MinibatchLoader.__index = MinibatchLoader

local function to_vector_list(l)
  for i = 1, #l do
    l[i] = l[i][{{},1}]
  end
  return l
end

function MinibatchLoader.create(opt, name)
  local self = {}
  setmetatable(self, MinibatchLoader)

  local data_file = path.join(opt.data_dir, name .. '.t7')

  print('loading data: ' .. name)
  local data = torch.load(data_file)
  
  self.attribute_batch_list = {}
  self.text_batch_list = {}
  local p = 0
  while p + opt.batch_size <= #data do
    -- bulid attribute matrix (3 is #attribute)
    local m_attr = torch.Tensor(opt.batch_size, 3)
    for i = 1, opt.batch_size do
      m_attr[i][1] = data[p + i][1]
      m_attr[i][2] = data[p + i][2]
      m_attr[i][3] = data[p + i][3]
    end
    table.insert(self.attribute_batch_list, to_vector_list(m_attr:split(1, 2)))
    -- build text matrix
    local max_len = #data[p + opt.batch_size][4]
    local m_text = torch.zeros(opt.batch_size, max_len + 2)
    -- add <S>
    m_text[{{}, 1}] = 1
    for i = 1, opt.batch_size do
      local w_list = data[p + i][4]
      for j = 1, #w_list do
        m_text[i][j + 1] = w_list[j]
      end
      -- add <E>
      m_text[i][#w_list + 2] = 2
    end
    table.insert(self.text_batch_list, m_text)

    p = p + opt.batch_size

    if p > 100000 then
      break
    end
  end

  -- reset batch index
  self.num_batch = #self.attribute_batch_list

  assert(#self.attribute_batch_list == #self.text_batch_list)

  collectgarbage()
  return self
end

function MinibatchLoader:random_batch()
  local p = math.random(self.num_batch)
  return self.attribute_batch_list[p], self.text_batch_list[p]
end

function MinibatchLoader:random_batch_list(n)
  local r = {}
  for i = 1, n do
    local p = math.random(self.num_batch)
    table.insert(r, {self.attribute_batch_list[p], self.text_batch_list[p]})
  end
  return r
end

function MinibatchLoader:all_batch()
  local r = {}
  for p = 1, self.num_batch do
    table.insert(r, {self.attribute_batch_list[p], self.text_batch_list[p]})
  end
  return r
end

return MinibatchLoader
