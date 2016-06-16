require 'image'
require 'nn'
util = paths.dofile('util.lua')
torch.setdefaulttensortype('torch.FloatTensor')

opt = {
	drop = 0,
	drag = 0,
	arithmetic = 'plus',  -- type of noise distribution (uniform / normal).
    net = '',              -- path to the generator network
    name = 'generation1',  -- name of the file saved
    gpu = 0,               -- gpu mode. 0 = CPU, 1 = GPU
    display = 1,           -- Display image: 0 = false, 1 = true
}
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
--print(opt)
if opt.display == 0 then opt.display = false end

assert(net ~= '', 'provide a generator model')

torch_path = "./../..//design_studio/static_files/torch/"

drop = torch.load(torch_path .. opt.drop .. ".bin")
drag = torch.load(torch_path .. opt.drag .. ".bin")

noise = ""
if opt.arithmetic == "plus" then
	noise = drop + drag
elseif opt.arithmetic == "minus" then
	noise = drop - drag

net = util.load(opt.net, opt.gpu)

-- for older models, there was nn.View on the top
-- which is unnecessary, and hinders convolutional generations.
if torch.type(net:get(1)) == 'nn.View' then
    net:remove(1)
end

if opt.gpu > 0 then
    require 'cunn'
    require 'cudnn'
    net:cuda()
    util.cudnn(net)
    noise = noise:cuda()
else
   net:float()
end

-- a function to setup double-buffering across the network.
-- this drastically reduces the memory needed to generate samples
util.optimizeInferenceMemory(net)

local images = net:forward(noise)
local cropImages = torch.Tensor(opt.batchSize, 3, 128, 42)

images:add(1):mul(0.5)

cropRange = 20;
for i = 1, opt.batchSize do
	startPatch = image.crop(images[i], 0, 64, cropRange, 64+cropRange)
	startIndex = 0
	threshold = torch.abs(startPatch:sum() - image.crop(images[i], 10, 64, 10+cropRange, 64+cropRange):sum()) + 5
	for j = 30, 48 do
		local patch = image.crop(images[i], j-cropRange, 64, j, 64+cropRange)
		startIndex = j
		if(torch.abs(startPatch:sum() - patch:sum()) > threshold) then
			break
		end
	end
	cropImages[i] = image.crop(images[i], startIndex, 0 , startIndex+42, 128)
end

generator_path = './../..//design_studio/static_files/generator/'
image.save(generator_path .. opt.name .. '.png', image.toDisplayTensor(cropImages))
print(opt.name .. '.png')

if opt.display then
    disp = require 'display'
    disp.image(cropImages)
end
