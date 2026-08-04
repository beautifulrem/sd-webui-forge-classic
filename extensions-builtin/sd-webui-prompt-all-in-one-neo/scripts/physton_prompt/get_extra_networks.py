# -*- coding: UTF-8 -*-

from modules import script_callbacks, extra_networks, prompt_parser, shared, ui_extra_networks
import json
import os
import copy
import ast

filters = [
    # 'filename',
    # 'description',
    'search_term',
    'local_preview',
    'metadata',
]


def _resolve_prompt_expression(expression):
    """Resolve Forge's generated Extra Networks prompt without JavaScript eval.

    Forge emits JSON string literals joined with ``+`` and may insert the one
    known ``opts.extra_networks_default_multiplier`` value. No names, calls,
    attributes, containers, or other operators are accepted.
    """
    if not isinstance(expression, str) or not expression.strip():
        return ''

    default_weight = getattr(shared.opts, 'extra_networks_default_multiplier', 1.0)
    normalized = expression.replace(
        'opts.extra_networks_default_multiplier',
        repr(default_weight),
    )
    try:
        tree = ast.parse(normalized, mode='eval')
    except (SyntaxError, ValueError):
        return ''

    def resolve(node):
        if isinstance(node, ast.Expression):
            return resolve(node.body)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return resolve(node.left) + resolve(node.right)
        if isinstance(node, ast.Constant) and isinstance(node.value, (str, int, float)):
            return str(node.value)
        raise ValueError('unsupported prompt expression')

    try:
        return resolve(tree)
    except ValueError:
        return ''


def get_extra_networks():
    result = []
    try:
        for extra_page in list(ui_extra_networks.extra_pages):
            result_item = {
                'name': extra_page.name,
                'title': extra_page.title,
                'items': []
            }
            for oriItem in extra_page.list_items():
                item = copy.deepcopy(oriItem)
                # 解析metadata
                output_name = None
                try:
                    if 'metadata' in item and item['metadata']:
                        metadata = json.loads(item['metadata'])
                        if metadata and 'ss_output_name' in metadata:
                            output_name = metadata['ss_output_name']
                except Exception as e:
                    pass
                item['output_name'] = output_name
                item['prompt_text'] = _resolve_prompt_expression(item.get('prompt', ''))

                # 获取civitai.info
                item['civitai_info'] = {}
                try:
                    if 'filename' in item and item['filename']:
                        item['basename'] = os.path.basename(item['filename'])
                        item['dirname'] = os.path.dirname(item['filename'])
                        base, ext = os.path.splitext(item['filename'])
                        info_file = base + '.civitai.info'
                        if not os.path.isfile(info_file):
                            info_file = item['filename'] + '.civitai.info'
                        if os.path.isfile(info_file):
                            with open(info_file, 'r') as f:
                                info = json.load(f)
                                images = info.get('images', [])
                                info = {
                                    'modelId': info.get('modelId', ''),
                                    'name': info.get('name', ''),
                                    'description': info.get('description', ''),
                                    'baseModel': info.get('baseModel', ''),
                                    'model': info.get('model', {}),
                                    'trainedWords': info.get('trainedWords', []),
                                    'images': [],
                                }
                                if images and len(images) > 0:
                                    for image in images:
                                        info['images'].append(image['url'])
                                item['civitai_info'] = info
                except Exception as e:
                    pass

                # 过滤掉不需要的字段
                newItem = {}
                for key in item:
                    if key not in filters:
                        newItem[key] = item[key]

                result_item['items'].append(newItem)

            result.append(result_item)
    except Exception as e:
        print(f'[sd-webui-prompt-all-in-one] get_extra_networks error: {e}')
        pass
    return result
