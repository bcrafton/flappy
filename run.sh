
# rm nohup.out;

# python flappy_bird.py --gpu 0 > results &
# python flappy_bird.py --gpu 1 > results1 &
# python flappy_bird.py --gpu 2 > results2 &

# python flappy_bird.py --gpu 0 > results3 &

# python flappy_bird.py --gpu 0 > results4 &
# python flappy_bird.py --gpu 1 > results5 &
# python flappy_bird.py --gpu 2 > results6 &
# python flappy_bird.py --gpu 3 > results7 &

# python flappy_bird_v2.py --gpu 0 > results8 &
# python flappy_bird_v2.py --gpu 1 > results9 &

# python flappy_bird_v1.py --gpu 2 > results10 &
# python flappy_bird_v2.py --gpu 3 > results11 &

python flappy_bird_v6.py --eps 0.00001 --lr 0.001   --gpu 0 > results0 &
python flappy_bird_v6.py --eps 0.00001 --lr 0.0003  --gpu 1 > results1 &
python flappy_bird_v6.py --eps 0.00001 --lr 0.0001  --gpu 2 > results2 &
python flappy_bird_v6.py --eps 0.00001 --lr 0.00003 --gpu 3 > results3 &

