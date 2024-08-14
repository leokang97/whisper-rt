from transcribe import transcribe
from utils import file_util
import constants


file_util.create_directory(constants.LOG_DIR_NAME)

transcribe.main()
