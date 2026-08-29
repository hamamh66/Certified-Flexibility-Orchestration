cd /home/user/paper
python3 campaign.py caseA 0 >> out/w0.log 2>&1
python3 campaign.py frontier 0 >> out/w0.log 2>&1
python3 campaign.py ablation 0 >> out/w0.log 2>&1
python3 campaign.py caseC 0 >> out/w0.log 2>&1
python3 campaign.py stress 0.0 >> out/w0.log 2>&1
python3 campaign.py stress 2.0 >> out/w0.log 2>&1
echo DONE0 >> out/w0.log
