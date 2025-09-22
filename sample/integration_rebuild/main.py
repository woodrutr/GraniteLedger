from pathlib import Path
from graniteledger.src.common.config_setup import Config_settings
from graniteledger.src.common.build import Build
#import graniteledger.src.models as Models
#import pyomo.environ as pyo
from graniteledger.src.common.Model import Model as model


def main():
    settings = Config_settings(
        config_path = Path()
    )

    meta = Build.build(settings)
#    meta.execute()

if __name__ == '__main__':
    main()