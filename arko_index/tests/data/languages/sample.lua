-- A tiny event emitter.
local Emitter = {}
Emitter.__index = Emitter

--- Creates an emitter.
function Emitter.new()
  local self = setmetatable({}, Emitter)
  self.handlers = {}
  return self
end

-- Registers a handler for an event.
function Emitter:on(event, handler)
  self.handlers[event] = self.handlers[event] or {}
  table.insert(self.handlers[event], handler)
end

function Emitter:emit(event, ...)
  for _, handler in ipairs(self.handlers[event] or {}) do
    handler(...)
  end
end

local function count(t)
  local n = 0
  for _ in pairs(t) do
    n = n + 1
  end
  return n
end

local helpers = {
  count = count,
  greet = function(name)
    return "hello " .. name
  end,
}

local e = Emitter.new()
e:on("ping", function() print("pong") end)
e:emit("ping")

return Emitter
