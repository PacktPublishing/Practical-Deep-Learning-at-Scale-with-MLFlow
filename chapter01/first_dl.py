import torch
import flash
from flash.core.data.utils import download_data
from flash.text import TextClassificationData, TextClassifier

print('### download IMDb data to local folder')
download_data("https://pl-flash-data.s3.amazonaws.com/imdb.zip", "./data/")
datamodule = TextClassificationData.from_csv(
    input_fields="review",
    target_fields="sentiment",
    train_file="data/imdb/train.csv",
    val_file="data/imdb/valid.csv",
    test_file="data/imdb/test.csv"
)

print('### define a text classifier ')
classifier_model = TextClassifier(backbone="prajjwal1/bert-tiny", num_classes=datamodule.num_classes)

print('### define the trainer')
trainer = flash.Trainer(max_epochs=3, gpus=torch.cuda.device_count())

print('### fine tune the pretrained model to get a new model for sentiment classification')
trainer.finetune(classifier_model, datamodule=datamodule, strategy="freeze")

print('### get prediction outputs for two sample sentences')
predictions = classifier_model.predict(
    [
        "Best movie I have seen.",
        "What a movie!",
    ]
)
print(predictions)

print('### get classifier test results')
trainer.test()
