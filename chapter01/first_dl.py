import torch
import flash
from flash.core.data.utils import download_data
from flash.text import TextClassificationData, TextClassifier

download_data("https://pl-flash-data.s3.amazonaws.com/imdb.zip", "./data/")
datamodule = TextClassificationData.from_csv(
    input_fields="review",
    target_fields="sentiment",
    train_file="data/imdb/train.csv",
    val_file="data/imdb/valid.csv",
    test_file="data/imdb/test.csv"
)

classifier_model = TextClassifier(backbone="prajjwal1/bert-tiny", num_classes=datamodule.num_classes)
trainer = flash.Trainer(max_epochs=3, gpus=torch.cuda.device_count())
trainer.finetune(classifier_model, datamodule=datamodule, strategy="freeze")

trainer.save_checkpoint("text_classification_model.pt")

predictions = classifier_model.predict(
    [
        "Best movie I have seen.",
        "The worst movie in the history of cinema.",
    ]
)

print(predictions)

trainer.test()
