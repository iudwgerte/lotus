from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("./tokenizer/lotus_tokenizer")
model = AutoModelForCausalLM.from_pretrained("./lotus_small")

multi_shot_prompt="""Find all c in Z_3 such that Z_3[x]/(x^2 + c) is a field.
A 0
B 1
C 2
D 3

Answer: B

Statement 1 | If aH is an element of a factor group, then |aH| divides |a|. Statement 2 | If H and K are subgroups of G then HK is a subgroup of G.
A True, True
B False, False
C True, False
D False, True

Answer: B

Statement 1 | Every element of a group generates a cyclic subgroup of the group. Statement 2 | The symmetric group S_10 has 10 elements.
A True, True
B False, False
C True, False
D False, True

Answer: C

Statement 1| Every function from a finite set onto itself must be one to one. Statement 2 | Every subgroup of an abelian group is abelian.
A True, True
B False, False
C True, False
D False, True

Answer: A

Find the characteristic of the ring 2Z.
A 0
B 3
C 12
D 30

Answer: A

{}
A {}
B {}
C {}
D {}

Answer: 
"""

def format_prompt(question, options):
    return multi_shot_prompt.format(question, options[0], options[1], options[2], options[3])
