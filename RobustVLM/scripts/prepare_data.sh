export base_dir=[/path/to/robust_vlm_data]
mkdir -p ${base_dir}
cd ${base_dir}

git clone https://huggingface.co/datasets/openflamingo/eval_benchmark

# coco train/val
cd ${base_dir}
mkdir coco
cd coco
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
unzip train2014.zip
unzip val2014.zip

# flickr30k
# get the Flickr 30k images from https://shannon.cs.illinois.edu/DenotationGraph/data/index.html
cd ${base_dir}
mkdir flickr30k; cd flickr30k
tar -xvf flickr30k-images.tar.gz
wget https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip
unzip caption_datasets.zip
ln -s dataset_flickr30k.json karpathy_flickr30k.json


# vizwiz
cd ${base_dir}
mkdir vizwiz; cd vizwiz
wget https://vizwiz.cs.colorado.edu/VizWiz_final/images/train.zip
wget https://vizwiz.cs.colorado.edu/VizWiz_final/images/val.zip
unzip train.zip
unzip val.zip

cd ${base_dir}
mkdir textvqa; cd textvqa
wget https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json
wget https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip
unzip train_val_images.zip