import torch
import torchaudio
import torchaudio.functional as F
import torch.nn.functional as nnF
from transformers import AutoConfig, Wav2Vec2Processor, AutoModelForAudioClassification


class RuEmotionClassifier:
    def __init__(self, model_name="KELONMYOSA/wav2vec2-xls-r-300m-emotion-ru", device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.config = AutoConfig.from_pretrained(model_name)
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.sampling_rate = self.processor.feature_extractor.sampling_rate

        self.model = AutoModelForAudioClassification.from_pretrained(
            model_name, trust_remote_code=True
        ).to(self.device)
        self.model.eval()

        self.our_classes = ['neutral', 'sad', 'angry', 'positive']

    def _load_waveform(self, audio_path):
        waveform, sr = torchaudio.load(audio_path)

        if sr != self.sampling_rate:
            waveform = F.resample(waveform, sr, self.sampling_rate)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        return waveform.squeeze(0).numpy()

    @torch.no_grad()
    def predict(self, audio_path):
        speech = self._load_waveform(audio_path)

        features = self.processor(
            speech, sampling_rate=self.sampling_rate, return_tensors="pt", padding=True
        )
        input_values = features.input_values.to(self.device)
        attention_mask = features.attention_mask.to(self.device)

        logits = self.model(input_values, attention_mask=attention_mask).logits
        scores = nnF.softmax(logits, dim=1).cpu().numpy()[0]

        all_scores = {self.config.id2label[i]: float(s) for i, s in enumerate(scores)}
        filtered = {k: v for k, v in all_scores.items() if k in self.our_classes}
        total = sum(filtered.values())
        normalized = {k: round(v / total, 5) for k, v in filtered.items()}

        predicted_label = max(normalized, key=normalized.get)

        return predicted_label, normalized