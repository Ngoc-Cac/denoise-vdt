import argparse, copy, pickle
import yaml, datasets
import torch

from torchmetrics.functional.text import word_error_rate
from transformers import TrainingArguments

from src.training_and_inference.trainer import DenoiserTrainer
from src.training_and_inference.training_utils import (
    load_vn_speech_transcript_datasets,
    load_dasheng,
    load_wav2vec2
)
from src.training_and_inference.nlp_utils import preprocess_vn_text


def parse_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-c", "--config_file",
        default='config.yaml'
    )

    return parser.parse_args()


with open("config.yaml") as file:
    train_config = yaml.safe_load(file)
torch.manual_seed(train_config['seed'])
device = 'cuda'


datasets = load_vn_speech_transcript_datasets(
    train_config['target_sr'],
    train_config['train-test-ratio'],
    return_data_loader=True,
    train_batch_size=train_config['batch_size'],
    test_batch_size=train_config['test_batch_size'],
    seed=train_config['seed']
)


model = load_dasheng().to(device)

if train_config['compute_wer'] or train_config['use_downstream_loss']:
    backbone = model.backbone.eval()
    istft_head = model.head.eval()

    processor, stt_model, downstream_loss_fn = load_wav2vec2()
    stt_model = stt_model.eval().to(device)

    def decode_embeddings(embeddings):
        return istft_head(backbone(embeddings))
else:
    decode_embeddings = None
    downstream_loss_fn = None

if not train_config['use_downstream_loss']:
    downstream_loss_fn = None

if train_config['compute_wer']:
    target_trans = [j for i in datasets['test']['loader'] for j in i[-1]]

    @torch.no_grad()
    def compute_metrics(eval_preds):
        preds = [
            processor.batch_decode(torch.argmax(stt_model(processor(
                wf[:lens],
                sampling_rate=16000,
                return_tensors="pt"
            ).input_values.to(device)).logits, dim=-1))[0]
            for wf, lens in zip(eval_preds.predictions, eval_preds.label_ids)
        ]

        return {'wer': word_error_rate(
            preprocess_vn_text(preds),
            preprocess_vn_text(target_trans)
        )}
else:
    compute_metrics = None


training_args = TrainingArguments(**train_config['trainer-args'])
trainer = DenoiserTrainer(
    model.feature_extractor.eval(),
    datasets['train']['loader'], datasets['test']['loader'],
    audio_decoder=decode_embeddings,
    downstream_loss_fn=downstream_loss_fn,
    downstream_loss_factor=.01,
    compute_metrics=compute_metrics,
    model_init=lambda: copy.deepcopy(model.second_encoder),
    args=training_args,
    eval_dataset=datasets['test']['speech_dataset']  # just so we can set eval_strategy
)

trainer.train()

if train_config['save_test_loader']:
    dest = f"{train_config['trainer-args']['output_dir']}/test-loader.pkl"
    with open(dest, 'wb') as file:
        pickle.dump(datasets['test']['loader']._cache, file)
