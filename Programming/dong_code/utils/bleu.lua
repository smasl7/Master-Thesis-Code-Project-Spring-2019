function count_ngram(x_list, n)
  local c = {}
  for i = 1, #x_list - n + 1 do
    -- get the ngram
    local k = ''
    for j = 1, n do
      k = k .. '-' .. tostring(x_list[i+j])
    end
    -- count
    if c[k] == nil then
      c[k] = 1
    else
      c[k] = c[k] + 1
    end
  end
  return c
end

--[[
input format:
list of word list
{
  {w0, w1, ..., w_n},
  {w0, w1, ..., w_n},
  ...
  {w0, w1, ..., w_n}
}
--]]
function compute_bleu(candidate_list, reference_list, N)
  if #candidate_list ~= #reference_list then
    print(string.format("BLEU: #candidate_list(%d) ~= #reference_list(%d)", #candidate_list, #reference_list))
  end

  local len = math.min(#candidate_list, #reference_list)
  local bleu_log = 0
  -- calculate modified precision
  local p = {}
  for n = 1, N do
    -- m: matched number
    local m = 0
    -- l: total number
    local l = 0

    -- smoothing
    if n ~= 1 then
      m = 1
      l = 1
    end

    for i = 1, len do
      if #candidate_list[i] >= n and #reference_list[i] >= n then
        candidate_ngram = count_ngram(candidate_list[i], n)
        reference_ngram = count_ngram(reference_list[i], n)
        -- count match
        for k, v in pairs(candidate_ngram) do
          if reference_ngram[k] ~= nil then
            m = m + math.min(candidate_ngram[k], reference_ngram[k])
          end
        end

        l = l + (#candidate_list[i] - n + 1)
      end
    end

    p[n] = m / l
  end

  for i = 1, N do
    bleu_log = bleu_log + math.log(p[i]) / N
  end

  -- calculate brevity penalty
  local r = 0
  local c = 0
  for i = 1, len do
    r = r + #reference_list[i]
    c = c + #candidate_list[i]
  end
  bleu_log = bleu_log + math.min(0, 1 - r / c)

  return math.exp(bleu_log)
end
