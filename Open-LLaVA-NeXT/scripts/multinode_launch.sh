# need to manually run this script on two nodes

VISION_TOWER=open_clip:hf-hub:zw123/delta_clip_h14_336
IMAGE_PROCESSOR_NAME_OR_PATH=[/path/to/Double_Visual_Defense/Open-LLaVA-NeXT/clip_preprocess/bv336/]
NNODES=1
NODE_RANK=0
MASTER_ADDR=[your master address]

################# adv #####################
PT_SCRIPT_PATH=[/path/to/Double_Visual_Defense/Open-LLaVA-NeXT/scripts_private/vlaa/v1_5/train/adv_pretrain_template_multinode.sh]
FT_SCRIPT_PATH=[/path/to/Double_Visual_Defense/Open-LLaVA-NeXT/scripts_private/vlaa/v1_5/train/adv_finetune_lora_template_multinode.sh]
#
## eps4 pgd-3/3 vitlr 1/20 base_lr
#EPSILON=0.01568627450980392
#STEP_SIZE=0.011764705882352941
#NUM_STEPS=3
#VISION_TOWER_LR=1e-6
#
## eps8 pgd-5/3 vitlr 1/20 base_lr
EPSILON=0.03137254901960784
STEP_SIZE=0.011764705882352941
NUM_STEPS=5
VISION_TOWER_LR=1e-6


## pretrain
bash $PT_SCRIPT_PATH $VISION_TOWER $PRETRAIN_VISION_TOWER $IMAGE_PROCESSOR_NAME_OR_PATH $EPSILON $STEP_SIZE $NUM_STEPS $NNODES $NODE_RANK $MASTER_ADDR
## finetune
bash $FT_SCRIPT_PATH $VISION_TOWER $PRETRAIN_VISION_TOWER $IMAGE_PROCESSOR_NAME_OR_PATH $EPSILON $STEP_SIZE $NUM_STEPS $VISION_TOWER_LR $NNODES $NODE_RANK $MASTER_ADDR $MM_PROJECTOR_PATH
