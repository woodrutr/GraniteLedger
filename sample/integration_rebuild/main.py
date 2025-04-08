from pathlib import Path
from src.common.config_setup import Config_settings
from src.common.build import Build
#import src.models as Models
#import pyomo.environ as pyo
from src.common.Model import Model as model


def main():
    settings = Config_settings(
        config_path = Path()
    )

    meta = Build.build(settings)
#    meta.execute()

if __name__ == '__main__':
    main()