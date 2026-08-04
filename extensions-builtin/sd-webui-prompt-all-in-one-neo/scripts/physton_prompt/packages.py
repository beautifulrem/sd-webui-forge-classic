import launch
from scripts.physton_prompt.get_lang import get_lang

packages = {
    "chardet": "chardet",
    "fastapi": "fastapi",
    "execjs": "PyExecJS",
    "lxml": "lxml",
    "tqdm": "tqdm",
    "pathos": "pathos",
    "cryptography": "cryptography",

    # The following packages are required for translation service. If you do not need translation service, you can remove them.
    # 以下是翻译所需的包，如果不需要翻译服务，可以删除掉它们。
    "openai": "openai",
    "boto3": "boto3",
    "aliyunsdkcore": "aliyun-python-sdk-core",
    "aliyunsdkalimt": "aliyun-python-sdk-alimt",
}


def get_packages_state():
    states = []
    for package_name in packages:
        package = packages[package_name]
        item = {
            'name': package_name,
            'package': package,
            'state': False
        }
        if launch.is_installed(package) or launch.is_installed(package_name):
            item['state'] = True

        states.append(item)

    return states


def install_package(name, package=None):
    result = {'state': False, 'message': ''}
    allowed_package = packages.get(name)
    # This function is reachable through an HTTP endpoint. Never pass a
    # caller-controlled requirement string to pip: only the fixed dependency
    # names declared above are installable. Keep checking ``package`` when it
    # is supplied so old clients cannot disguise a different requirement
    # behind an allowed display name.
    if allowed_package is None or (package is not None and package != allowed_package):
        result['message'] = f'Package is not allowed: {name}'
        return result

    try:
        launch.run_pip(
            f"install {allowed_package}",
            f"sd-webui-prompt-all-in-one: {name}",
        )
        result['state'] = True
        result['message'] = get_lang('install_success', {'0': allowed_package})
    except Exception as e:
        print(e)
        print(f'Warning: Failed to install {allowed_package}, some preprocessors may not work.')
        result['message'] = get_lang('install_failed', {'0': allowed_package}) + '\n' + str(e)
    return result
