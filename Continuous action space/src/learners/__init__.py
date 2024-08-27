from .cq_learner import CQLearner
from .facmac_learner import FACMACLearner
from .facmac_learner_discrete import FACMACDiscreteLearner
from .maddpg_learner import MADDPGLearner
from .maddpg_learner_discrete import MADDPGDiscreteLearner
from .q_adv_learner import QLearner as QADVLearner
from .q_adv_learner_discrete import QLearner as QADVLearner_discrete

REGISTRY = {}
REGISTRY["cq_learner"] = CQLearner
REGISTRY["facmac_learner"] = FACMACLearner
REGISTRY["facmac_learner_discrete"] = FACMACDiscreteLearner
REGISTRY["maddpg_learner"] = MADDPGLearner
REGISTRY["maddpg_learner_discrete"] = MADDPGDiscreteLearner
REGISTRY["q_adv_learner"] = QADVLearner

REGISTRY["q_adv_learner_discrete"] = QADVLearner_discrete