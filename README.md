1. create content data
   1. create images from fonts ```python font.py --fonts fonts/JP_Ronde-B_square.otf fonts/JP_NotoSerifJP-Regular.otf fonts/JP_ReggaeOne-Regular.ttf fonts/JP_saruji.ttf fonts/JP_TsunagiGothic.ttf fonts/JP_yokomoji.otf --charset=JP --sample_dir=content_dir```
   3. create binary files ```python package.py --dir=content_dir --save_dir=experiment/content_data```
2. create style data
    1. create images from fonts ```python font.py --font=fonts/JP_Ronde-B_square.otf --charset=JP --sample_dir=style_dir --char_num=5```
    2. create binary files ```python package.py --dir=style_dir --save_dir=experiment/style_data```
4. train ```python train.py --style_input_nc=5 --content_input_nc=5 --experiment_dir=experiment --content_data_dir=experiment/content_data --style_data_dir=experiment/style_data --epoch=1000 --batch_size=16 --schedule=100 --sample_steps=5 --check_point_steps=100 gpu_id=cuda```
