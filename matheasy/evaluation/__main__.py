"""Check run logs against the required fields: python -m matheasy.evaluation runs/"""

import sys

from .log_fields import main

sys.exit(main(sys.argv[1:]))
