import torch
import torch.nn.init as init
import torch.nn as nn
import torchaudio.functional as F
import torch.nn.functional as nnF
from transformers import ASTModel, ASTConfig, ASTFeatureExtractor


class EmoModelAST(nn.Module):
    """
    Изменения входа по сравнению с VGG:
    VGG принимал mel-спектрограмму как "картинку" (80 mel-полос, torchaudio.transforms.MelSpectrogram).
    AST принимает специфичный формат fbank-признаков
      (128 mel-полос — фиксировано архитектурой предобученной модели),
      посчитанный через ASTFeatureExtractor. Использовать наш mel-препроцессинг
      нельзя: веса AST обучены на конкретный способ извлечения признаков,
      несовпадение формата сломает transfer learning.
      time_frames паддится до 1024 (соответствует ~10.24 сек
      входного аудио) — под это обучены позиционные эмбеддинги модели

    Классификация (classifier) сохранена в том же виде, что и в VGG-версии
    (Linear -> ReLU -> Dropout -> Linear), меняется только размер входа
    (768 — hidden_size AST, вместо 256*2 у VGG).

    Заморозка слоёв (freeze_first_n_layers):
    Заморозил половину 6/12 слоев, мне показалось этого будет достаточно для дообучения для эмоций

    """
    def __init__(self, num_classes=4, pretrained="MIT/ast-finetuned-audioset-10-10-0.4593", freeze_first_n_layers=6):
        super().__init__()
        self.ast = ASTModel.from_pretrained(pretrained)
        hidden_size = self.ast.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.35),
            nn.Linear(512, num_classes),
        )

        self._initialize_weights()

        for p in self.ast.embeddings.parameters():
            p.requires_grad = False
        for layer in self.ast.encoder.layer[:freeze_first_n_layers]:
            for p in layer.parameters():
                p.requires_grad = False


    def _initialize_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                init.normal_(m.weight, 0, 0.01)
                init.constant_(m.bias, 0)

    def forward(self, input_values):
        outputs = self.ast(input_values=input_values)
        pooled = outputs.pooler_output
        x = self.classifier(pooled)
        return x

    """ 
    Чтобы не раздувать кодовую базу API,
    решил прописать тут инференс
    """

    def _get_features_extractor(self):
        if not hasattr(self, '_features_extractor'):
            self._features_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

        return self._features_extractor

    @torch.no_grad()
    def predict_from_waveform(self, waveform, sample_rate, target_sample_rate=16000,
                               emotion_labels=('neutral', 'sad', 'angry', 'positive')):
        device = next(self.parameters()).device

        if sample_rate != target_sample_rate:
            waveform = F.resample(waveform, sample_rate, target_sample_rate)

        if waveform.dim() == 2 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if waveform.dim() == 2:
            waveform = waveform.squeeze(0)

        feature_extractor = self._get_features_extractor()
        inputs = feature_extractor(
            waveform.numpy(),
            sampling_rate=target_sample_rate,
            return_tensors="pt",
        )
        input_values = inputs["input_values"].to(device)

        logits = self.forward(input_values)
        probs = nnF.softmax(logits, dim=1).cpu().numpy()[0]

        scores = {label: round(float(p), 5) for label, p in zip(emotion_labels, probs)}
        predicted_label = max(scores, key=scores.get)

        return predicted_label, scores