# onmt = {}
#
# require('onmt.modules.init')
#
# onmt.data = require('onmt.data.init')
# onmt.train = require('onmt.train.init')
# onmt.translate = require('onmt.translate.init')
# onmt.utils = require('onmt.utils.init')
#
# onmt.Constants = require('onmt.Constants')
# onmt.Models = require('onmt.Models')

import onmt.Constants
import onmt.Models
from onmt.Dataset import Dataset, collate_data
from onmt.Optim import Optim

# return onmt
