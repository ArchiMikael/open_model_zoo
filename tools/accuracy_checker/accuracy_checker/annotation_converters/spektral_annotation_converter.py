from ..representation import ClassificationAnnotation
from pathlib import Path
from .format_converter import GraphFileBasedAnnotationConverter, ConverterReturn
from spektral.utils.io import load_binary


class SpektralConverter(GraphFileBasedAnnotationConverter):
    __provider__ = "spektral_converter"

    def convert(self, check_content=False, **kwargs):
        graph = load_binary(Path(self.graph_path).__str__())

        labels = graph.y

        annotation = [
            ClassificationAnnotation(identifier='', label=labels)
        ]

        return ConverterReturn(annotation, {'labels': labels}, None)