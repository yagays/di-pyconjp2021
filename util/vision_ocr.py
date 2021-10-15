import json
from dataclasses import dataclass, astuple
from typing import List

from PIL import Image, ImageDraw
import proto


@dataclass
class Vertex:
    x: int
    y: int


@dataclass
class BoundingBox:
    top_left: Vertex
    top_right: Vertex
    bottom_left: Vertex
    bottom_right: Vertex


@dataclass
class Symbol:
    text: str
    confidence: float
    bounding_box: BoundingBox
    property: dict


@dataclass
class Word:
    confidence: float
    bounding_box: List[Vertex]
    symbols: List[Symbol]

    @property
    def text(self):
        return "".join([s.text for s in self.symbols])

    @property
    def detectedbreak(self):
        for symbol in self.symbols:
            if "detectedBreak" in symbol.property:
                return symbol.property["detectedBreak"]["type"]
        return ""


@dataclass
class Paragraph:
    confidence: float
    bounding_box: BoundingBox
    words: List[Word]

    @property
    def splitted_text(self):
        output = []
        words_text = ""
        for word in self.words:
            if word.detectedbreak:
                words_text += word.text
                output.append(words_text)
                words_text = ""
            else:
                words_text += word.text
        return output

    @property
    def text(self):
        return "".join([w.text for w in self.words])

    def __str__(self):
        return f"<Paragraph> {self.text}"


@dataclass
class Block:
    confidence: float
    bounding_box: BoundingBox
    paragraphs: List[Paragraph]
    block_type: int


@dataclass
class Page:
    blocks: List[Block]
    width: int
    height: int
    property: dict


@dataclass
class Line:
    bounding_box: BoundingBox
    words: List[Word]
    text: str


def vertices2bounding_box(vertex):
    return BoundingBox(
        top_left=Vertex(vertex[0].x, vertex[0].y),
        top_right=Vertex(vertex[1].x, vertex[1].y),
        bottom_right=Vertex(vertex[2].x, vertex[2].y),
        bottom_left=Vertex(vertex[3].x, vertex[3].y),
    )


def proto_message_to_dict(message: proto.Message) -> dict:
    """Helper method to parse protobuf message to dictionary."""
    return json.loads(message.__class__.to_json(message))


class VisionOCR:
    def __init__(self, response, image):
        self.all_blocks = []
        self.all_paragraphs = []
        self.all_words = []
        self.all_symbols = []
        self.all_pages = []

        if isinstance(image, str):
            # file_path
            self.image = Image.open(image)
        else:
            # ndarray
            self.image = Image.fromarray(image)
        self._analyze_ocr_result(response.full_text_annotation)
        self.full_text = response.full_text_annotation.text
        self.detectedbreak_type2text = {
            0: "UNKNOWN",
            1: "SPACE",
            2: "SURE_SPACE",
            3: "EOL_SURE_SPACE",
            4: "HYPHEN",
            5: "LINE_BREAK",
        }
        self.all_lines = self._get_all_line()

    def _analyze_ocr_result(self, document):
        pages = []
        for page in document.pages:
            blocks = []
            for block in page.blocks:
                paragraphs = []
                for paragraph in block.paragraphs:
                    words = []
                    for word in paragraph.words:
                        symbols = []
                        for symbol in word.symbols:

                            s = Symbol(
                                text=symbol.text,
                                confidence=symbol.confidence,
                                bounding_box=vertices2bounding_box(symbol.bounding_box.vertices),
                                property=proto_message_to_dict(symbol.property),
                            )
                            symbols.append(s)

                        w = Word(
                            confidence=word.confidence,
                            symbols=symbols,
                            bounding_box=vertices2bounding_box(word.bounding_box.vertices),
                        )
                        words.append(w)
                        self.all_symbols += symbols

                    p = Paragraph(
                        confidence=paragraph.confidence,
                        words=words,
                        bounding_box=vertices2bounding_box(paragraph.bounding_box.vertices),
                    )
                    paragraphs.append(p)
                    self.all_words += words

                b = Block(
                    confidence=block.confidence,
                    paragraphs=paragraphs,
                    bounding_box=vertices2bounding_box(block.bounding_box.vertices),
                    block_type=block.block_type,
                )
                blocks.append(b)
                self.all_paragraphs += paragraphs

            pa = Page(
                blocks=blocks,
                width=page.width,
                height=page.height,
                property=proto_message_to_dict(page.property),
            )
            pages.append(pa)
            self.all_blocks = blocks
        self.all_pages += pages

    def get_all_paragraph_text(self):
        return [p.text for p in self.all_paragraphs]

    def get_all_word_text(self):
        return [w.text for w in self.all_words]

    def get_all_symbol_text(self):
        return [s.text for s in self.all_symbols]

    def draw_hint(self, feature_type, file_name, show_box_number=False):
        im = self.image.copy()
        draw = ImageDraw.Draw(im)
        if feature_type == "block":
            self._draw_rectangle(draw, self.all_blocks, "red", show_box_number)
        elif feature_type == "paragraph":
            self._draw_rectangle(draw, self.all_paragraphs, "blue", show_box_number)
        elif feature_type == "word":
            self._draw_rectangle(draw, self.all_words, "green", show_box_number)
        elif feature_type == "symbol":
            self._draw_rectangle(draw, self.all_symbols, "pink", show_box_number)
        elif feature_type == "line":
            self._draw_rectangle(draw, self.all_lines, "cyan", show_box_number)
        else:
            raise TypeError("feature_type can be only block/paragraph/word/symbol/line")

        return im

    def _draw_rectangle(self, draw, all_boxes, color, show_box_number=False, show_box_annotation=False):
        for i, box in enumerate(all_boxes):
            draw.rectangle(
                [
                    astuple(box.bounding_box.top_left),
                    astuple(box.bounding_box.bottom_right),
                ],
                outline=color,
            )

            if isinstance(box, Word) and box.detectedbreak and show_box_annotation:
                for symbol in box.symbols:
                    if "detectedBreak" in symbol.property:
                        draw.rectangle(
                            [
                                astuple(symbol.bounding_box.top_left),
                                astuple(symbol.bounding_box.bottom_right),
                            ],
                            outline="red",
                        )
                        self._draw_box_annotation(
                            draw,
                            symbol.bounding_box,
                            self.detectedbreak_type2text[symbol.property["detectedBreak"]["type"]],
                            "red",
                        )

            if show_box_number:
                self._draw_box_annotation(draw, box.bounding_box, str(i), color)

    def _draw_box_annotation(self, draw, bounding_box, text, color):
        text_w, text_h = draw.textsize(text)
        draw.rectangle(
            [
                astuple(bounding_box.top_left),
                (bounding_box.top_left.x + text_w, bounding_box.top_left.y - text_h),
            ],
            outline=color,
            fill=color,
        )
        draw.text(
            (bounding_box.top_left.x, bounding_box.top_left.y - text_h),
            text,
            fill="white",
        )

    def stats(self):
        return {
            "num_blocks": len(self.all_blocks),
            "num_paragraphs": len(self.all_paragraphs),
            "num_words": len(self.all_words),
            "num_symbols": len(self.all_symbols),
            "num_lines": len(self.all_lines),
        }

    def export_box_as_image(self, feature_type, output_path):
        if feature_type == "block":
            target_list = self.all_blocks
        elif feature_type == "paragraph":
            target_list = self.all_paragraphs
        elif feature_type == "word":
            target_list = self.all_words
        elif feature_type == "symbol":
            target_list = self.all_symbols
        elif feature_type == "line":
            target_list = self.all_lines
        else:
            raise TypeError("feature_type can be only block/paragraph/word/symbol/line")

        im = self.image
        for i, symbol in enumerate(target_list):
            im.crop(
                (
                    symbol.bounding_box.top_left.x,
                    symbol.bounding_box.top_left.y,
                    symbol.bounding_box.bottom_right.x,
                    symbol.bounding_box.bottom_right.y,
                )
            ).save(f"{output_path}/{feature_type}_{i}.png")

    def _get_all_line(self, strict=False):
        output = []

        line_list = self._split_word_sequence_by_detectedbreak()
        for word_list in line_list:
            output.append(self._merge_word_sequence(word_list))

        if strict:
            output = self._strict_line_merge(output)

        return output

    def merge_splitted_line(self, strict=True):
        self.all_lines = self._get_all_line(strict=strict)

    def _strict_line_merge(self, line_list, overlap_threshold=0.6):
        """merge lines within the same y coordinates"""
        output = []
        merged_line = [line_list[0]]
        for line in line_list[1:]:
            left_line_top_y = merged_line[-1].bounding_box.top_right.y
            left_line_bottom_y = merged_line[-1].bounding_box.bottom_right.y
            right_line_top_y = line.bounding_box.top_left.y
            right_line_bottom_y = line.bounding_box.bottom_left.y

            height = max(
                0,
                min(left_line_bottom_y, right_line_bottom_y) - max(left_line_top_y, right_line_top_y),
            )

            bottom_y = max(left_line_bottom_y, right_line_bottom_y)
            top_y = min(left_line_top_y, right_line_top_y)

            if height > (bottom_y - top_y) * overlap_threshold:
                merged_line.append(line)
            else:
                output.append(merged_line)
                merged_line = [line]

        output.append(merged_line)
        return [self._merge_line_sequence(o) for o in output]

    def _merge_line_sequence(self, line_list):
        left_corner_line = line_list[0]
        right_corner_line = line_list[-1]
        merged_words = []
        merged_text = ""
        for line in line_list:
            merged_words += line.words
            merged_text += line.text

        return Line(
            bounding_box=BoundingBox(
                top_left=left_corner_line.bounding_box.top_left,
                top_right=right_corner_line.bounding_box.top_right,
                bottom_left=left_corner_line.bounding_box.bottom_left,
                bottom_right=right_corner_line.bounding_box.bottom_right,
            ),
            words=merged_words,
            text=merged_text,
        )

    def _merge_word_sequence(self, word_list):
        left_corner_word = word_list[0]
        right_corner_word = word_list[-1]
        return Line(
            bounding_box=BoundingBox(
                top_left=left_corner_word.bounding_box.top_left,
                top_right=right_corner_word.bounding_box.top_right,
                bottom_left=left_corner_word.bounding_box.bottom_left,
                bottom_right=right_corner_word.bounding_box.bottom_right,
            ),
            words=word_list,
            text="".join([w.text for w in word_list]),
        )

    def _split_word_sequence_by_detectedbreak(self):
        output = []
        line = []
        for word in self.all_words:
            line.append(word)
            if word.detectedbreak and self.detectedbreak_type2text[word.detectedbreak] in (
                "EOL_SURE_SPACE",
                "LINE_BREAK",
            ):
                output.append(line)
                line = []
        return output
