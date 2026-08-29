cd /home/user/paper
while ! (grep -q DONE0 out/w0.log 2>/dev/null && grep -q DONE1 out/w1.log 2>/dev/null); do sleep 30; done
python3 weights.py >> out/w2.log 2>&1
python3 runtime.py >> out/w2.log 2>&1
echo DONE2 >> out/w2.log
