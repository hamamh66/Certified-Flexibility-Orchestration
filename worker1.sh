cd /home/user/paper
python3 campaign.py caseA 1 >> out/w1.log 2>&1
python3 campaign.py frontier 1 >> out/w1.log 2>&1
python3 campaign.py ablation 1 >> out/w1.log 2>&1
python3 campaign.py caseC 1 >> out/w1.log 2>&1
python3 campaign.py stress 0.5 >> out/w1.log 2>&1
python3 campaign.py stress 3.0 >> out/w1.log 2>&1
python3 campaign.py stress 1.0 >> out/w1.log 2>&1
echo DONE1 >> out/w1.log
