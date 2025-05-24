from super_gradients.training import Trainer, models, datasets
from super_gradients.training.utils.detection_utils import DetectionMetrics
from super_gradients.training.transforms import DetectionMosaic

# 1. Trainer 생성
trainer = Trainer(experiment_name="yolo_nas_test", ckpt_root_dir="checkpoints")

# 2. 데이터셋 파라미터
dataset_path = "/home/aistore02/yolo-nas-aistore/project_aistore"

common_dataset_params = {
    "data_dir": dataset_path,
    "input_dim": (640, 480),
    "classes": ['aunt_jemima_original_syrup', 'band_aid_clear_strips', 'mahatma_rice', 'white_rain_body_wash', 'pringles_bbq', 'cheeze_it', 'hersheys_bar', 'redbull', 'mom_to_mom_sweet_potato_corn_apple', 'a1_steak_sauce', 'jif_creamy_peanut_butter', 'cinnamon_toast_crunch', 'bumblebee_albacore', 'arm_hammer_baking_soda', 'dr_pepper', 'haribo_gold_bears_gummi_candy', 'bulls_eye_bbq_sauce_original', 'reeses_pieces', 'clif_crunch_peanut_butter', 'mom_to_mom_butternut_squash_pear', 'pop_tararts_strawberry', 'quaker_big_chewy_chocolate_chip', 'spam', 'cholula_chipotle_hot_sauce', 'coffee_mate_french_vanilla', 'pepperidge_farm_milk_chocolate_macadamia_cookies', 'kitkat_king_size', 'snickers', 'toblerone_milk_chocolate', 'clif_z_bar_chocolate_chip', 'nature_valley_crunchy_oats_n_honey', 'ritz_crackers', 'palmolive_orange', 'crystal_hot_sauce', 'crayola_24_crayons', 'tapatio_hot_sauce', 'nabisco_nilla_wafers', 'pepperidge_farm_milano_cookies_double_chocolate', 'campbells_chicken_noodle_soup', 'frappuccino_coffee', 'chewy_dips_chocolate_chip', 'chewy_dips_peanut_butter', 'nature_vally_fruit_and_nut', 'cheerios', 'lindt_excellence_cocoa_dark_chocolate', 'hersheys_cocoa', 'hersheys_symphony', 'campbells_chunky_classic_chicken_noodle', 'martinellis_apple_juice', 'dove_pink', 'dove_white', 'david_sunflower_seeds', 'monster_energy', 'act_ii_butter_lovers_popcorn', 'coca_cola_glass_bottle', 'twix', 'honey_bunches_of_oats_honey_roasted', 'honey_bunches_of_oats_with_almonds', 'hunts_sauce', 'listerine_green'],  # 클래스 리스트
    "cache": False,
}
mosaic_transforms = DetectionMosaic(
    input_dim=(640, 480),
    mosaic_probability=0.8
)

train_data = datasets.get(
    dataset_name="yolo_nas",
    dataset_params={
        **common_dataset_params,
        "images_dir": "images/train",
        "labels_dir": "labels/train",
        "transforms": mosaic_transforms,
    }
)

val_data = datasets.get(
    dataset_name="yolo_nas",
    dataset_params={
        **common_dataset_params,
        "images_dir": "images/val",
        "labels_dir": "labels/val",
    }
)

# 3. COCO pretrained YOLO-NAS 모델 불러오기
model = models.get("yolo_nas_s", num_classes=60, pretrained_weights="coco")

# 4. 학습 시작
trainer.train(
    model=model,
training_params={
    "max_epochs": 100,
    "batch_size": 128,
    "initial_lr": 0.01,
    "lr_mode": "cosine",
    "cosine_final_lr_ratio": 0.01,
    "loss": "yolo_nas_loss",
    "optimizer": "AdamW",  # or "SGD" with momentum
    "optimizer_params": {
        "weight_decay": 1e-4
    },
    "train_metrics_list": [DetectionMetrics()],
    "valid_metrics_list": [DetectionMetrics()],
    "mixed_precision": True,
    "average_best_models": True,
    "experiment_name": "yolo_nas",
    "loggers": ["wandb"],

}
,
    train_loader=train_data,
    valid_loader=val_data
)
