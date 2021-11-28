from pkg_resources import resource_filename

import wine_platform


def get_raw_path():
    return resource_filename(wine_platform.__name__, "data/raw")
