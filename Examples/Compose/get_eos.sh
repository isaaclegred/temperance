EOSPATH=$1
EOSLABEL=$2
wget -O ${EOSLABEL}_thermo.csv ${EOSPATH}/eos.thermo
wget -O ${EOSLABEL}_nb.csv ${EOSPATH}/eos.nb
wget -O ${EOSLABEL}_t.csv ${EOSPATH}/eos.t
wget -O ${EOSLABEL}_yq.csv ${EOSPATH}/eos.yq
