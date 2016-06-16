require 'image'
torch.setdefaulttensortype('torch.FloatTensor')

opt = {
	uid = "",
	bestIndex = 1
}
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end

torch_path = "./../..//design_studio/static_files/torch/"
file = torch_path .. opt.uid .. ".bin"

noise = torch.load(file)
best = noise[opt.bestIndex]:reshape(1, 75, 1, 1)

torch.save(file, best)
