local MaskedClassNLLCriterion, parent = torch.class('MaskedClassNLLCriterion', 'nn.Criterion')

function MaskedClassNLLCriterion:__init()
  parent.__init(self)
  self.sizeAverage = true
  self.outputTensor = torch.Tensor(1)
end

function MaskedClassNLLCriterion:__len()
  return 0
end

function MaskedClassNLLCriterion:updateOutput(input, target)
  if input:type() == 'torch.CudaTensor' then
    if self.weights == nil then
      -- The CUDA implementation requires self.weights be non-nil
      self.weights = torch.CudaTensor()
    end
    assert(self.weights:dim() == 0 or self.weights:dim() == 1,
      'weights must be 1D or empty')
    -- The cuda code wont check weight size, so we must do it here.
    if self.weights:dim() == 1 then
      if input:dim() == 1 then
        assert(self.weights:size(1) == input:size(1), 'Wrong number of weights')
      else
        assert(self.weights:size(1) == input:size(2), 'Wrong number of weights')
      end
    end

    if input:dim() == 1 then
      self._target = self._target or input.new(1)
      if type(target) == 'number' then
        self._target[1] = target
      else
        self._target:copy(target)
      end
      input.nn.ClassNLLCriterion_updateOutput(self, input, self._target)
    else
      input.nn.ClassNLLCriterion_updateOutput(self, input, target)
    end
    self.output = self.outputTensor[1]
    return self.output
  end

  if input:dim() == 1 then
    if torch.isTensor(target) then 
      target = target[1] 
    end
    -- mask
    if target == 0 then
      self.output = 0
    else
      self.output = -input[target]
    end
  else
    if input:dim() == 2 then
      local output = 0
      for i=1,target:size(1) do
        -- mask
        if target[i] ~= 0 then
          output = output - input[i][target[i]]
        end
      end
      if self.sizeAverage then
        output = output / target:size(1)
      end
      self.output = output
    else
      error('matrix or vector expected')
    end
  end
  return self.output
end

function MaskedClassNLLCriterion:updateGradInput(input, target)
  self.gradInput:resizeAs(input)
  self.gradInput:zero()
  if input:type() == 'torch.CudaTensor' then
    if input:dim() == 1 then
      self._target = self._target or input.new(1)
      if type(target) == 'number' then
        self._target[1] = target
      else
        self._target:copy(target)
      end
      input.nn.ClassNLLCriterion_updateGradInput(self, input, self._target)
    else
      input.nn.ClassNLLCriterion_updateGradInput(self, input, target)
    end
    return self.gradInput
  end

  if input:dim() == 1 then
    if torch.isTensor(target) then 
      target = target[1] 
    end
    -- mask
    if target ~= 0 then
      self.gradInput[target] = -1
    end
  else
    local z = -1
    if self.sizeAverage then
      z = z / target:size(1)
    end
    for i=1,target:size(1) do
      -- mask
      if target[i] ~= 0 then
        self.gradInput[i][target[i]] = z
      end
    end
  end
  return self.gradInput
end
