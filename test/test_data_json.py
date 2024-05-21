import jsonlines

data_dir = '/ML-A100/team/mm/yanghuan/data/pexels_clip'
train_meta = '/ML-A100/team/mm/zixit/data/pexles_20240405/pexels_meta_train.jsonl'
video_list = []
caption_list = []
with open(train_meta, 'r', encoding="utf-8") as f:
    cnt = 0
    for line in jsonlines.Reader(f):
        # print(line)
        video_list.append(line['clip_id']+'.mp4')
        caption_list.append(line['llava_medium'])
        cnt += 1
        if cnt == 2:
            break

print(video_list)
print(caption_list)
assert len(video_list) == len(caption_list)
