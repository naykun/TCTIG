
import json
import os
import pickle
import torch
import numpy as np
import collections
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union
from omegaconf import OmegaConf

from abc import ABC
import lmdb
from sample import Sample
import random
from transformers.models.auto import AutoTokenizer
from utils import byte_tensor_to_object, object_to_byte_tensor
from PIL import Image
import torchvision
import torchvision.datasets.folder as tv_helpers
from torchvision import transforms
# from mmf.datasets.databases.readers.feature_readers import FeatureReader

class TimedPoint(NamedTuple):
    x: float
    y: float
    t: float


class TimedUtterance(NamedTuple):
    utterance: str
    start_time: float
    end_time: float


class LocalizedNarrative(NamedTuple):
    dataset_id: str
    image_id: str
    annotator_id: int
    caption: str
    timed_caption: Optional[List[TimedUtterance]] = None
    traces: Optional[List[List[TimedPoint]]] = None
    voice_recording: Optional[str] = None

    def __repr__(self):
        truncated_caption = (
            self.caption[:60] + "..." if len(self.caption) > 63 else self.caption
        )
        truncated_timed_caption = self.timed_caption[0].__str__()
        truncated_traces = self.traces[0][0].__str__()
        return (
            f"{{\n"
            f" dataset_id: {self.dataset_id},\n"
            f" image_id: {self.image_id},\n"
            f" annotator_id: {self.annotator_id},\n"
            f" caption: {truncated_caption},\n"
            f" timed_caption: [{truncated_timed_caption}, ...],\n"
            f" traces: [[{truncated_traces}, ...], ...],\n"
            f" voice_recording: {self.voice_recording}\n"
            f"}}"
        )


class LocalizedNarrativesAnnotationDatabase(torch.utils.data.Dataset):
    def __init__(self, config, path, *args, **kwargs):
        super().__init__()
        self.config = config
        self.path = path
        self.start_idx = 0
        self.load_annotation_db(path)

    def load_annotation_db(self, path):
        # import ipdb; ipdb.set_trace()
        data = []
        if path.endswith(".jsonl"):
            self.store_type = "jsonl"
            with open(path) as f:
                for line in f:
                    annotation = json.loads(line)
                    loc_narr = LocalizedNarrative(**annotation)
                    data.append(
                        {
                            "dataset_id": loc_narr.dataset_id,
                            "image_id": loc_narr.image_id,
                            "caption": loc_narr.caption,
                            "feature_path": self._feature_path(
                                loc_narr.dataset_id, loc_narr.image_id
                            ),
                            "timed_caption": loc_narr.timed_caption,
                            "traces": loc_narr.traces,
                        }
                    )
        elif path.endswith(".lmdb"):
            self.store_type = "lmdb"
            self.lmdb_path = path
            self.lmdb_env = None
            env = lmdb.open(
                path,
                subdir=os.path.isdir(path),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
            with env.begin(write=False, buffers=True) as txn:
                data = list(pickle.loads(txn.get(b"keys")))
        self.data = data

    def init_env(self):
        self.lmdb_env = lmdb.open(
            self.lmdb_path,
            subdir=os.path.isdir(self.lmdb_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

    def __getitem__(self, idx):
        data = self.data[idx + self.start_idx]
        if self.store_type == "lmdb":
            if self.lmdb_env is None:
                self.init_env()
            with self.lmdb_env.begin(write=False, buffers=True) as txn:
                data = pickle.loads(txn.get(data))
                loc_narr = LocalizedNarrative(**data)
                data = {
                    "dataset_id": loc_narr.dataset_id,
                    "image_id": loc_narr.image_id,
                    "caption": loc_narr.caption,
                    "feature_path": self._feature_path(
                        loc_narr.dataset_id, loc_narr.image_id
                    ),
                    "timed_caption": loc_narr.timed_caption,
                    "traces": loc_narr.traces,
                }
                # import ipdb; ipdb.set_trace()
        return data

    def _feature_path(self, dataset_id, image_id):
        if "mscoco" in dataset_id.lower():
            return image_id.rjust(12, "0") + ".npy"

        return image_id + ".npy"

    def __len__(self):
        return len(self.data)


def get_possible_image_paths(path):
    path = os.path.abspath(path)
    image_path = path.split(".")
    # Image path might contain file extension (e.g. .jpg),
    # In this case, we want the path without the extension
    image_path = image_path if len(image_path) == 1 else image_path[:-1]
    for ext in tv_helpers.IMG_EXTENSIONS:
        image_ext = ".".join(image_path) + ext
        if os.path.isfile(image_ext):
            path = image_ext
            break
    # import ipdb; ipdb.set_trace()
    return path


def default_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class ImageDatabase(torch.utils.data.Dataset):
    """ImageDatabase can be used to load images in MMF.
    This goes either in conjunction with AnnotationDatabase or
    can be separately used with function such as `from_path`.
    MMFDataset initializes its own copy of ImageDatabase if `use_images`
    is True. Rest everything works same as a normal torch Dataset if
    you pass the annotation_db as a parameter. For example for item
    1 from annotation db, you can pass same id to ImageDatabase to loads
    its image. If you don't pass it, you have two options. Either use
    .get which takes in an annotation db item or .from_path which directly
    takes in an image path. You are free to use your own dataset instead
    of image database or free to update or ignore MMFDataset's ImageDataset
    initialization. You can either reinitialize with transform and other
    params or use any of torchvision's datasets.
    """

    def __init__(
        self,
        config,
        path,
        annotation_db=None,
        transform=None,
        loader=default_loader,
        is_valid_file=None,
        image_key=None,
        *args,
        **kwargs
    ):
        """Initialize an instance of ImageDatabase

        Args:
            torch ([type]): [description]
            config (DictConfig): Config object from dataset_config
            path (str): Path to images folder
            annotation_db (AnnotationDB, optional): Annotation DB to be used
                to be figure out image paths. Defaults to None.
            transform (callable, optional): Transform to be called upon loaded image.
                Defaults to None.
            loader (callable, optional): Custom loader for image which given a path
                returns a PIL Image. Defaults to torchvision's default loader.
            is_valid_file (callable, optional): Custom callable to filter out invalid
                files. If image is invalid, {"images": []} will returned which you can
                filter out in your dataset. Defaults to None.
            image_key (str, optional): Key that points to image path in annotation db.
                If not specified, ImageDatabase will make some intelligent guesses
                about the possible key. Defaults to None.
        """
        super().__init__()
        self.config = config
        self.base_path = path
        self.transform = transform
        self.annotation_db = annotation_db
        self.loader = loader
        self.image_key = config.get("image_key", None)
        self.image_key = image_key if image_key else self.image_key
        self.is_valid_file = is_valid_file

    @property
    def annotation_db(self):
        return self._annotation_db

    @annotation_db.setter
    def annotation_db(self, annotation_db):
        self._annotation_db = annotation_db

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, transform):
        if isinstance(transform, collections.abc.MutableSequence):
            transform = torchvision.Compose(transform)
        self._transform = transform

    def __len__(self):
        self._check_annotation_db_present()
        return len(self.annotation_db)

    def __getitem__(self, idx):
        self._check_annotation_db_present()
        item = self.annotation_db[idx]
        return self.get(item)

    def _check_annotation_db_present(self):
        if not self.annotation_db:
            raise AttributeError(
                "'annotation_db' must be set for the database to use __getitem__."
                + " Use image_database.annotation_db to set it."
            )

    def get(self, item):
        possible_images = self._get_attrs(item)
        return self.from_path(possible_images)

    def from_path(self, paths, use_transforms=True):
        if isinstance(paths, str):
            paths = [paths]

        assert isinstance(
            paths, collections.abc.Iterable
        ), "Path needs to a string or an iterable"

        loaded_images = []
        for image in paths:
            image = os.path.join(self.base_path, image)
            path = get_possible_image_paths(image)

            valid = self.is_valid_file(path) if self.is_valid_file is not None else True

            if not valid:
                continue

            if not path:
                # Create the full path without extension so it can be printed
                # for the error
                possible_path = ".".join(image.split(".")[:-1])

                raise RuntimeError(
                    "Image not found at path {}.{{jpeg|jpg|svg|png}}.".format(
                        possible_path
                    )
                )
            image = self.open_image(path)

            if self.transform and use_transforms:
                image = self.transform(image)
            loaded_images.append(image)

        return {"images": loaded_images}

    def open_image(self, path):
        return self.loader(path)

    def _get_attrs(self, item):
        """Returns possible attribute that can point to image id

        Args:
            item (Object): Object from the DB

        Returns:
            List[str]: List of possible images that will be copied later
        """
        if self.image_key:
            image = item[self.image_key]
            if isinstance(image, str):
                image = [image]
            return image

        image = None
        pick = None
        attrs = self._get_possible_attrs()

        for attr in attrs:
            image = item.get(attr, None)
            if image is not None:
                pick = attr
                break

        if pick == "identifier" and "left_url" in item and "right_url" in item:
            return [image + "-img0", image + "-img1"]
        else:
            return [image]

    def _get_possible_attrs(self):
        return [
            "Flickr30kID",
            "Flikr30kID",
            "identifier",
            "image_path",
            "image_name",
            "img",
            "image_id",
        ]



class CVLGLocalizedNarrativesDataset(torch.utils.data.Dataset):

    def __init__(self, config, dataset_type="train", index=0) -> None:
        super().__init__()
        self.dataset_type = dataset_type
        self._index = index
        self.config = config
        self.annotation_db = self.build_annotation_db()
        self.caption_processor = TracedBertTokenizer(config.processors.caption_processor.params)
        self.trace_bbox_processor = SpatialTraceTokenizer(config.processors.trace_bbox_processor.params)
        self.image_processor = TorchvisionTransforms(self.config.processors.image_processor.params)

        self._use_images = self.config.get("use_images", False)
        if self._use_images:
            self.image_db = self.build_image_db()

        # self._use_features = self.config.get("use_features", False)
        # if self._use_features:
        #     self.features_db = self.build_features_db()

    def _get_path_based_on_index(self, config, attribute):
        if attribute not in config:
            raise ValueError(f"{attribute} not present in config")

        config = config.get(attribute, None)

        if (
            self.dataset_type not in config
            or len(config.get(self.dataset_type, [])) == 0
        ):
            raise ValueError(f"No {attribute} present for type {self.dataset_type}")

        paths = config[self.dataset_type]

        if isinstance(paths, str):
            selected_path = paths
        else:
            assert isinstance(paths, collections.abc.MutableSequence)
            selected_path = paths[self._index]

        selected_path = self._add_root_dir(selected_path)

        return selected_path

    def _add_root_dir(self, path):
        path = path.split(",")
        for idx, p in enumerate(path):
            path[idx] = os.path.join(self.config.data_dir, p)

        return ",".join(path)

    def build_annotation_db(self) -> LocalizedNarrativesAnnotationDatabase:
        annotation_path = self._get_path_based_on_index(
            self.config, "annotations"
        )
        return LocalizedNarrativesAnnotationDatabase(self.config, annotation_path)

    # def build_features_db(self):
    #     features_path = self._get_path_based_on_index(
    #         self.config, "features"
    #     )
    #     return FeaturesDatabase(
    #         self.config, features_path, annotation_db=self.annotation_db
    #     )

    def build_image_db(self):
        image_path = self._get_path_based_on_index(self.config, "images")
        return ImageDatabase(self.config, image_path, annotation_db=self.annotation_db)

    def __getitem__(self, idx: int) -> Sample:
        sample_info = self.annotation_db[idx]
        current_sample = Sample()

        if self._use_images:
            # import ipdb; ipdb.set_trace()
            image_id = sample_info["image_id"]
            dataset = sample_info["dataset_id"]
            if "mscoco" in dataset:
                image_id = image_id.rjust(12, "0")

            assert (
                len(self.image_db.from_path(image_id)["images"]) != 0
            ), f"image id: {image_id} not found"
            image = self.image_db.from_path(image_id)["images"][0]
            current_sample.image = self.image_processor(image)
        # breakpoint()

        processed_caption = self.caption_processor(
            {
                "timed_caption": sample_info["timed_caption"]
            }
        )
        # should be a trace enhanced processor
        current_sample.update(processed_caption)
        # print(processed_caption.get("sync_reverse",False))
        processed_traces = self.trace_bbox_processor(
            sample_info,
            processed_caption.get("sync_reverse", False),
            processed_caption.get("sync_shuffle_order", None),
        )
        current_sample.update(processed_traces)
        current_sample.image_id = object_to_byte_tensor(sample_info["image_id"])
        current_sample.feature_path = sample_info["feature_path"]
        # import ipdb; ipdb.set_trace()

        return current_sample

    def format_for_prediction(self, report):
        captions = report.captions.tolist()
        cross_attentions = report.cross_attention.tolist()
        predictions = []

        for idx, image_id in enumerate(report.image_id):
            image_id = byte_tensor_to_object(image_id)
            cross_attention = cross_attentions[idx]
            caption = self.caption_processor.id2tokens(captions[idx]).split()
            raw_caption = self.caption_processor.id2rawtoken(captions[idx])
            if isinstance(image_id, torch.Tensor):
                image_id = image_id.item()
            predictions.append(
                {
                    "image_id": image_id,
                    "caption": caption,
                    "cross_attention": cross_attention,
                    "raw_caption": raw_caption,
                }
            )

        return predictions

    def __len__(self):
        return len(self.annotation_db)


class MaskedTokenProcessor():
    _CLS_TOKEN = "[CLS]"
    _SEP_TOKEN = "[SEP]"
    _MASK_TOKEN = "[MASK]"
    _PAD_TOKEN_ID = 0

    def __init__(self, config, *args, **kwargs):

        tokenizer_config = config.tokenizer_config
        self._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_config.type, **tokenizer_config.params
        )

        self._max_seq_length = config.max_seq_length
        self._probability = getattr(config, "mask_probability", 0.15)

    def get_vocab_size(self) -> int:
        return len(self._tokenizer)

    def tokenize(self, tokens: Union[str, List[str]]) -> List[str]:
        return self._tokenizer.tokenize(tokens)

    def _convert_tokens_to_ids(
        self, tokens: Union[str, List[str]]
    ) -> Union[int, List[int]]:
        return self._tokenizer.convert_tokens_to_ids(tokens)

    def _convert_ids_to_tokens(
        self, ids: Union[int, List[int]]
    ) -> Union[str, List[str]]:
        return self._tokenizer.convert_ids_to_tokens(ids)

    def _random_word(
        self, tokens: List[str], probability: float = 0.15
    ) -> Tuple[List[str], List[int]]:
        labels = []
        for idx, token in enumerate(tokens):
            prob = random.random()

            if prob < probability:
                prob /= probability

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[idx] = self._MASK_TOKEN
                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[idx] = self._convert_ids_to_tokens(
                        torch.randint(self.get_vocab_size(), (1,), dtype=torch.long)
                    )[0]

                # rest 10% keep the original token as it is

                labels.append(self._convert_tokens_to_ids(token))
            else:
                labels.append(-1)

        return tokens, labels

    def _truncate_seq_pair(
        self, tokens_a: List[str], tokens_b: List[str], max_length: int
    ):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        if tokens_b is None:
            tokens_b = []
            max_length -= 2
        else:
            # _convert_to_indices does [CLS] tokens_a [SEP] tokens_b [SEP]
            max_length -= 3
        assert max_length >= 0, (
            "Max length should be minimum 2 in case of single sentence"
            + " and 3 in case of two sentences."
        )

        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()




class TracedBertTokenizer(MaskedTokenProcessor):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self._probability = 0
        self.segment_reverse = (
            config.segment_reverse if hasattr(config, "segment_reverse") else False
        )
        self.sync_seg_reverse = (
            config.sync_seg_reverse if hasattr(config, "sync_seg_reverse") else False
        )
        self.sync_seg_shuffle = (
            config.sync_seg_shuffle if hasattr(config, "sync_seg_shuffle") else False
        )

    def __call__(self, item):

        timed_caption = item["timed_caption"]
        tokens = []
        seg_ids = None
        # print(bbox_attend_scores.shape)
        # print(len(timed_caption))

        seg_indices = []
        for i, word in enumerate(timed_caption):
            text = word["utterance"]
            # wired length mis-matching
            token = self.tokenize(text)
            tokens += token
            if "." in text:
                seg_indices.append(len(tokens))
        # guard for sentence without "."
        if len(seg_indices) == 0 or seg_indices[-1] != len(tokens):
            seg_indices.append(len(tokens))

        if self.sync_seg_reverse:
            import random

            sync_reverse = random.random() > 0.5

        if self.segment_reverse or (self.sync_seg_reverse and sync_reverse):
            seg_start = [0] + seg_indices[:-1]
            seg_end = seg_indices
            seg_s_e = list(zip(seg_start, seg_end))
            # print(seg_s_e)
            tokens_segs = [tokens[s:e] for s, e in seg_s_e]

            if self.sync_seg_shuffle:
                shuffle_order = list(range(len(tokens_segs)))
                random.shuffle(shuffle_order)
                tokens_segs = [tokens_segs[i] for i in shuffle_order]
            else:
                tokens_segs.reverse()

            tokens = [token for seg in tokens_segs for token in seg]
            # start from 1
            seg_ids = [i + 1 for i, seg in enumerate(tokens_segs) for token in seg]
            # import ipdb; ipdb.set_trace()

        tokens = tokens[: self._max_seq_length - 1]
        if seg_ids is None:
            seg_ids = [1] * len(tokens)
        seg_ids = seg_ids[: self._max_seq_length - 1]

        output = self._convert_to_indices(
            tokens, seg_ids,
        )
        if self.sync_seg_reverse:
            output["sync_reverse"] = sync_reverse
            if self.sync_seg_shuffle and sync_reverse:
                output["sync_shuffle_order"] = shuffle_order
            else:
                output["sync_shuffle_order"] = None
        return output

    def _convert_to_indices(self, tokens, seg_ids):
        tokens = [self._CLS_TOKEN] + tokens
        # attend_length = len(token_attends[0])
        segment_ids = [0] + seg_ids

        input_ids = self._tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < self._max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == self._max_seq_length
        assert len(input_mask) == self._max_seq_length
        assert len(segment_ids) == self._max_seq_length

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "input_mask": input_mask,
            "segment_ids": segment_ids,
            "text": tokens,
        }

    def id2tokens(self, ids):
        return self._tokenizer.decode(ids, skip_special_tokens=True)

    def id2rawtoken(self, ids):
        return self._tokenizer.convert_ids_to_tokens(ids)



class SpatialTraceTokenizer():
    def __init__(self, config, *args, **kwargs):
        self._max_seq_length = config.max_seq_length
        self.delta = config.delta
        self.reverse = config.reverse
        self.segment_reverse = (
            config.segment_reverse if hasattr(config, "segment_reverse") else False
        )
        self.sync_seg_reverse = (
            config.sync_seg_reverse if hasattr(config, "sync_seg_reverse") else False
        )
        self.time_window = config.time_window if config.time_window else 0.4
        # import ipdb; ipdb.set_trace()

    def __call__(
        self, sample_info, sync_reverse=False, sync_shuffle_order=None
    ):
        traces = [x for tr in sample_info["traces"] for x in tr]

        current_t = 0
        current_trace_window = []
        trace_boxes = []
        for t in traces:
            if t["t"] > current_t:
                current_t += self.time_window
                if len(current_trace_window) > 0:
                    points = np.array(current_trace_window)
                    x1, y1 = points.min(axis=0) * (1 - self.delta)
                    x2, y2 = points.max(axis=0) * (1 + self.delta)
                    area = (x2 - x1) * (y2 - y1)
                    trace_boxes.append([x1, y1, x2, y2, area, t["t"]])
                    current_trace_window = []
            else:
                current_trace_window.append([t["x"], t["y"]])
        if self.segment_reverse or (self.sync_seg_reverse and sync_reverse):
            timed_caption = sample_info["timed_caption"]
            time_slot = []
            for utter in timed_caption:
                if "." in utter["utterance"]:
                    time_slot.append(utter["end_time"])
            segments = []
            segment = []
            seg_id = 0
            for box in trace_boxes:
                if seg_id < len(time_slot) and box[-1] > time_slot[seg_id]:
                    seg_id += 1
                    segments.append(segment)
                    segment = []
                else:
                    segment.append(box[:-1])
            if len(segment) > 0:
                segments.append(segment)
                segment = []
            if sync_shuffle_order is not None:
                max_segments_id = len(segments) - 1
                # print(len_segments)
                # print(sync_shuffle_order)
                if max_segments_id >= 0:
                    segments = [
                        segments[min(i, max_segments_id)] for i in sync_shuffle_order
                    ]
            else:
                segments.reverse()
            trace_boxes = [box for seg in segments for box in seg]
            seg_id = [i + 1 for i, seg in enumerate(segments) for box in seg]
        else:
            trace_boxes = [box[:-1] for box in trace_boxes]
            seg_id = [1] * len(trace_boxes)
        trace_boxes, trace_boxes_mask, boxes_seg_id, contr_seg_id = self._trancate(
            trace_boxes, seg_id
        )
        trace_boxes = torch.tensor(trace_boxes, dtype=torch.float)
        trace_boxes_mask = torch.tensor(trace_boxes_mask, dtype=torch.long)
        boxes_seg_id = torch.tensor(boxes_seg_id, dtype=torch.long)
        contr_seg_id = torch.tensor(contr_seg_id, dtype=torch.long)
        return {
            "trace_boxes": trace_boxes,
            "trace_boxes_mask": trace_boxes_mask,
            "trace_boxes_seg_id": boxes_seg_id,
            "trace_boxes_loop_contrastive_seg_id": contr_seg_id,
        }

    def _trancate(self, boxes, seg_id):
        boxes = boxes[: self._max_seq_length]
        seg_id = seg_id[: self._max_seq_length]
        if self.reverse and not self.segment_reverse:
            boxes.reverse()
        num_boxes = len(boxes)
        appendix = [[0.0] * 5] * (self._max_seq_length - num_boxes)
        boxes += appendix
        box_mask = [1] * num_boxes + [0] * (self._max_seq_length - num_boxes)
        loop_contrastive_seg_id = seg_id + [0] * (self._max_seq_length - num_boxes)
        box_seg_id = [1] * self._max_seq_length
        return boxes, box_mask, box_seg_id, loop_contrastive_seg_id


class TorchvisionTransforms():
    def __init__(self, config, *args, **kwargs):
        transform_params = config.transforms
        assert OmegaConf.is_dict(transform_params) or OmegaConf.is_list(
            transform_params
        )
        if OmegaConf.is_dict(transform_params):
            transform_params = [transform_params]

        transforms_list = []
        # import ipdb; ipdb.set_trace()
        for param in transform_params:
            if OmegaConf.is_dict(param):
                # This will throw config error if missing
                transform_type = param.type
                transform_param = param.get("params", OmegaConf.create({}))
            else:
                assert isinstance(param, str), (
                    "Each transform should either be str or dict containing "
                    + "type and params"
                )
                transform_type = param
                transform_param = OmegaConf.create([])

            transform = getattr(transforms, transform_type, None)
            if transform is None:
                if transform_type == "GrayScaleTo3Channels":
                    transform = GrayScaleTo3Channels
            # https://github.com/omry/omegaconf/issues/248
            transform_param = OmegaConf.to_container(transform_param)
            # If a dict, it will be passed as **kwargs, else a list is *args
            if isinstance(transform_param, collections.abc.Mapping):
                transform_object = transform(**transform_param)
            else:
                transform_object = transform(*transform_param)

            transforms_list.append(transform_object)

        self.transform = transforms.Compose(transforms_list)

    def __call__(self, x):
        # Support both dict and normal mode
        if isinstance(x, collections.abc.Mapping):
            x = x["image"]
            return {"image": self.transform(x)}
        else:
            return self.transform(x)

class GrayScaleTo3Channels():
    def __init__(self, *args, **kwargs):
        return

    def __call__(self, x):
        if isinstance(x, collections.abc.Mapping):
            x = x["image"]
            return {"image": self.transform(x)}
        else:
            return self.transform(x)

    def transform(self, x):
        assert isinstance(x, torch.Tensor)
        # Handle grayscale, tile 3 times
        if x.size(0) == 1:
            x = torch.cat([x] * 3, dim=0)
        return x





if __name__=="__main__":
    from utils import load_yaml
    config = load_yaml("./src/config/caption_coco2017.yaml")
    # print(config.dataset_config.cvlg_coco2017)
    from dalle_pytorch import OpenAIDiscreteVAE, VQGanVAE1024
    from tqdm import tqdm
    import pickle
    # vae = OpenAIDiscreteVAE().cuda()
    vae = VQGanVAE1024().cuda()
    for subset in ["train", "val"]:
        dataset = CVLGLocalizedNarrativesDataset(config.dataset_config.cvlg_coco2017, dataset_type=subset)
        print(len(dataset))
        results = {}
        for idx, sample in enumerate(tqdm(dataset, total=len(dataset))):
            code = vae.get_codebook_indices(sample["image"].unsqueeze(0).cuda())[0]
            import ipdb; ipdb.set_trace()
            result = {"idx":idx, 
                "image_code":code.detach().cpu().tolist(),
                "input_ids": sample.input_ids.tolist(),
                "input_mask": sample.input_mask.tolist(),
                "segment_ids": sample.segment_ids.tolist(),
                "trace_boxes": sample.trace_boxes.tolist(),
                "trace_boxes_mask": sample.trace_boxes_mask.tolist(),
                "trace_boxes_seg_ids": sample.trace_boxes_seg_id.tolist(),
                "trace_boxes_loop_contrastive_seg_id": sample.trace_boxes_loop_contrastive_seg_id.tolist(),
                "image_id":sample.feature_path.split(".")[0],
                "text": sample.text
            }
            # print(result)
            # print(code)
            # print(code.shape)
            # print(sample.keys())
            # print(sample.trace_boxes)
            # print(sample.text)
            results[idx] = result
            # print(sample.input_ids)
            # break
        pickle.dump(results,open(os.path.join("data",subset+"_vqgan_1024.data"),"wb"))
        result_load = pickle.load(open(os.path.join("data",subset+"_vqgan_1024.data"),"rb"))
        assert len(result_load) == len(results)