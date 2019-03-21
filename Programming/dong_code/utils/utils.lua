require "torch"
require "nn"
require "nngraph"
require "optim"

function combine_all_parameters(...)
    --[[ like module:getParameters, but operates on many modules ]]--

    -- get parameters
    local networks = {...}
    local parameters = {}
    local gradParameters = {}
    for i = 1, #networks do
        local net_params, net_grads = networks[i]:parameters()

        if net_params then
            for _, p in pairs(net_params) do
                parameters[#parameters + 1] = p
            end
            for _, g in pairs(net_grads) do
                gradParameters[#gradParameters + 1] = g
            end
        end
    end

    local function storageInSet(set, storage)
        local storageAndOffset = set[torch.pointer(storage)]
        if storageAndOffset == nil then
            return nil
        end
        local _, offset = unpack(storageAndOffset)
        return offset
    end

    -- this function flattens arbitrary lists of parameters,
    -- even complex shared ones
    local function flatten(parameters)
        if not parameters or #parameters == 0 then
            return torch.Tensor()
        end
        local Tensor = parameters[1].new

        local storages = {}
        local nParameters = 0
        for k = 1,#parameters do
            local storage = parameters[k]:storage()
            if not storageInSet(storages, storage) then
                storages[torch.pointer(storage)] = {storage, nParameters}
                nParameters = nParameters + storage:size()
            end
        end

        local flatParameters = Tensor(nParameters):fill(1)
        local flatStorage = flatParameters:storage()

        for k = 1,#parameters do
            local storageOffset = storageInSet(storages, parameters[k]:storage())
            parameters[k]:set(flatStorage,
                storageOffset + parameters[k]:storageOffset(),
                parameters[k]:size(),
                parameters[k]:stride())
            parameters[k]:zero()
        end

        local maskParameters=  flatParameters:float():clone()
        local cumSumOfHoles = flatParameters:float():cumsum(1)
        local nUsedParameters = nParameters - cumSumOfHoles[#cumSumOfHoles]
        local flatUsedParameters = Tensor(nUsedParameters)
        local flatUsedStorage = flatUsedParameters:storage()

        for k = 1,#parameters do
            local offset = cumSumOfHoles[parameters[k]:storageOffset()]
            parameters[k]:set(flatUsedStorage,
                parameters[k]:storageOffset() - offset,
                parameters[k]:size(),
                parameters[k]:stride())
        end

        for _, storageAndOffset in pairs(storages) do
            local k, v = unpack(storageAndOffset)
            flatParameters[{{v+1,v+k:size()}}]:copy(Tensor():set(k))
        end

        if cumSumOfHoles:sum() == 0 then
            flatUsedParameters:copy(flatParameters)
        else
            local counter = 0
            for k = 1,flatParameters:nElement() do
                if maskParameters[k] == 0 then
                    counter = counter + 1
                    flatUsedParameters[counter] = flatParameters[counter+cumSumOfHoles[k]]
                end
            end
            assert (counter == nUsedParameters)
        end
        return flatUsedParameters
    end

    -- flatten parameters and gradients
    local flatParameters = flatten(parameters)
    local flatGradParameters = flatten(gradParameters)

    -- return new flat vector that contains all discrete parameters
    return flatParameters, flatGradParameters
end

--[[ Creates clones of the given network.
The clones share all weights and gradWeights with the original network.
Accumulating of the gradients sums the gradients properly.
The clone also allows parameters for which gradients are never computed
to be shared. Such parameters must be returns by the parametersNoGrad
method, which can be null.
--]]
function cloneManyTimes(net, T)
  local clones = {}
  local params, gradParams = net:parameters()
  if params == nil then
    params = {}
  end
  local paramsNoGrad
  if net.parametersNoGrad then
    paramsNoGrad = net:parametersNoGrad()
  end
  local mem = torch.MemoryFile("w"):binary()
  mem:writeObject(net)
  for t = 1, T do
    -- We need to use a new reader for each clone.
    -- We don't want to use the pointers to already read objects.
    local reader = torch.MemoryFile(mem:storage(), "r"):binary()
    local clone = reader:readObject()
    reader:close()
    local cloneParams, cloneGradParams = clone:parameters()
    local cloneParamsNoGrad
    for i = 1, #params do
      cloneParams[i]:set(params[i])
      cloneGradParams[i]:set(gradParams[i])
    end
    if paramsNoGrad then
      cloneParamsNoGrad = clone:parametersNoGrad()
      for i =1,#paramsNoGrad do
        cloneParamsNoGrad[i]:set(paramsNoGrad[i])
      end
    end
    clones[t] = clone
    collectgarbage()
  end
  mem:close()
  return clones
end

function str_hash(str)
  local hash = 1
  for i = 1, #str, 2 do
    hash = math.fmod(hash * 12345, 452930459) +
    ((string.byte(str, i) or (len - i)) * 67890) +
    ((string.byte(str, i + 1) or i) * 13579)
  end
  return hash
end

function init_device(opt)
  if opt.gpuid >= 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then print('package cunn not found!') end
    if not ok2 then print('package cutorch not found!') end
    if ok and ok2 then
        print('using CUDA on GPU ' .. opt.gpuid .. '...')
        cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        cutorch.manualSeed(opt.seed)
    else
        print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
        print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
        print('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
  end

  torch.manualSeed(opt.seed)
end

function copy_table(to, from)
  assert(#to == #from)
  for i = 1, #to do
    to[i]:copy(from[i])
  end
end

function add_table(to, from)
  assert(#to == #from)
  for i = 1, #to do
    to[i]:add(from[i])
  end
end

function clone_table(from)
  local to = {}
  for i = 1, #from do
    table.insert(to, from[i]:clone())
  end
  return to
end

function shallowcopy(orig)
    local orig_type = type(orig)
    local copy
    if orig_type == 'table' then
        copy = {}
        for orig_key, orig_value in pairs(orig) do
            copy[orig_key] = orig_value
        end
    else -- number, string, boolean, etc
        copy = orig
    end
    return copy
end

function table_topk(t, k)
  local r = {}
  for i = 1, math.min(#t, k) do
    table.insert(r, t[i])
  end
  return r
end

function argmax(vector)
  if vector:dim() == 1 then
    local v_max = vector:max()
    for i = 1, vector:size(1) do
      if vector[i] == v_max then
        return i
      end
    end
  else
    error("Argmax only supports vectors")
  end
end

floor = torch.floor
ceil = torch.ceil
random = torch.random
