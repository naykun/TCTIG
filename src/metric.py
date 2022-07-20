import torch
from dataset_raw import TracedBertTokenizer


class TracedCaptionBleu4Metric():
    """Metric for calculating caption accuracy using BLEU4 Score.

    **Key:** ``caption_bleu4``
    """

    def __init__(self, config):
        import nltk.translate.bleu_score as bleu_score

        self._bleu_score = bleu_score
        self.caption_processor = TracedBertTokenizer(config.tokenizer)
        self.required_params = ["captions", "input_ids"]
    
    def __call__(self, *args, **kwargs):
        return self.calculate(*args, **kwargs)

    def calculate(self, sample_list, model_output, *args, **kwargs):
        """Calculate accuracy and return it back.

        Args:
            sample_list (SampleList): SampleList provided by DataLoader for
                                current iteration
            model_output (Dict): Dict returned by model.

        Returns:
            torch.FloatTensor: bleu4 score.

        """
        # Create reference and hypotheses captions.
        references = []
        hypotheses = []

        # References
        targets = sample_list.input_ids.tolist()
        for j, _ in enumerate(targets):
            img_captions = [self.caption_processor.id2tokens(targets[j]).split()]
            references.append(img_captions)

        # Hypotheses
        if "captions" in model_output:
            scores = model_output["captions"]
        else:
            scores = torch.max(model_output["scores"], dim=-1)[1]
        scores = scores.tolist()
        predictions = []
        for j, _ in enumerate(scores):
            caption = self.caption_processor.id2tokens(scores[j]).split()
            predictions.append(caption)
        hypotheses.extend(predictions)

        assert len(references) == len(hypotheses)
        # breakpoint()
        bleu4 = self._bleu_score.corpus_bleu(references, hypotheses)

        return sample_list.input_ids.new_tensor(bleu4, dtype=torch.float)

