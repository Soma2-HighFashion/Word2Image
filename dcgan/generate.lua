require 'image'
require 'nn'
util = paths.dofile('util.lua')
torch.setdefaulttensortype('torch.FloatTensor')

opt = {
    batchSize = 1,        -- number of samples to produce
    noisetype = 'normal',  -- type of noise distribution (uniform / normal).
    net = '',              -- path to the generator network
    imsize = 1,            -- used to produce larger images. 1 = 64px. 2 = 80px, 3 = 96px, ...
    noisemode = 'random',  -- random / line / linefull1d / linefull
    name = 'generation1',  -- name of the file saved
    gpu = 1,               -- gpu mode. 0 = CPU, 1 = GPU
    display = 1,           -- Display image: 0 = false, 1 = true
    nz = 75,              
}
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
--print(opt)
if opt.display == 0 then opt.display = false end

assert(net ~= '', 'provide a generator model')

noise = torch.Tensor(opt.batchSize, opt.nz, opt.imsize, opt.imsize)
net = util.load(opt.net, opt.gpu)

-- for older models, there was nn.View on the top
-- which is unnecessary, and hinders convolutional generations.
if torch.type(net:get(1)) == 'nn.View' then
    net:remove(1)
end

--print(net)

if opt.noisetype == 'uniform' then
    noise:uniform(-1, 1)
elseif opt.noisetype == 'normal' then
    noise:normal(0, 1)
end

noiseL = torch.FloatTensor(opt.nz):uniform(-1, 1)
noiseR = torch.FloatTensor(opt.nz):uniform(-1, 1)
if opt.noisemode == 'line' then
   -- do a linear interpolation in Z space between point A and point B
   -- each sample in the mini-batch is a point on the line
    line  = torch.linspace(0, 1, opt.batchSize)
    for i = 1, opt.batchSize do
        noise:select(1, i):copy(noiseL * line[i] + noiseR * (1 - line[i]))
    end
elseif opt.noisemode == 'linefull1d' then
   -- do a linear interpolation in Z space between point A and point B
   -- however, generate the samples convolutionally, so a giant image is produced
    assert(opt.batchSize == 1, 'for linefull1d mode, give batchSize(1) and imsize > 1')
    noise = noise:narrow(3, 1, 1):clone()
    line  = torch.linspace(0, 1, opt.imsize)
    for i = 1, opt.imsize do
        noise:narrow(4, i, 1):copy(noiseL * line[i] + noiseR * (1 - line[i]))
    end
elseif opt.noisemode == 'linefull' then
   -- just like linefull1d above, but try to do it in 2D
    assert(opt.batchSize == 1, 'for linefull mode, give batchSize(1) and imsize > 1')
    line  = torch.linspace(0, 1, opt.imsize)
    for i = 1, opt.imsize do
        noise:narrow(3, i, 1):narrow(4, i, 1):copy(noiseL * line[i] + noiseR * (1 - line[i]))
    end
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

--print('images size: ', images:size(1)..' x '..images:size(2) ..' x '..images:size(3)..' x '..images:size(4))
images:add(1):mul(0.5)

for i = 1, opt.batchSize do
	startPatch = image.crop(images[i], 0, 0, 5, 5)
	upper = startPatch:mean()+0.16; lower = startPatch:mean()-0.16
	startIndex = 0
	for j = 3, 48 do
		local patch = image.crop(images[i], j-3, 0, j, 3)
		startIndex = j
		if(patch:mean() > upper or patch:mean() < lower) then
			break
		end
	end
	print('Images size: ', images[i]:size(1)..' x '..images[i]:size(2) ..' x '..images[i]:size(3))
	print(startIndex)
	cropImages[i] = image.crop(images[i], startIndex-2, 0 , startIndex+40, 128)
end
--print('Min, Max, Mean, Stdv', images:min(), images:max(), images:mean(), images:std())
print('cropImages size: ', cropImages:size(1)..' x '..cropImages:size(2) ..' x '..cropImages:size(3)..' x '..cropImages:size(4))

--generator_path = '/home/dj/HighFashionProject/design_studio/static_files/generator/'
generator_path = './../..//design_studio/static_files/generator/'
image.save(generator_path .. opt.name .. '.png', image.toDisplayTensor(cropImages))
print(opt.name .. '.png')

if opt.display then
    disp = require 'display'
    disp.image(cropImages)
    --print('Displayed image')
end
