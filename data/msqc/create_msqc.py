import json

with open("test.txt", 'w') as test_file:
    for l in open("test_public_v1.1.json"):
        jl = json.loads(l)
        line = str(jl['query_type']).upper() +":"+ jl['query'] + " ?\n";
        test_file.write(line)

with open("train.txt", 'w') as test_file:
    for l in open("train_v1.1.json"):
        jl = json.loads(l)
        line = str(jl['query_type']).upper() +":"+ jl['query'] + " ?\n";
        test_file.write(line)
