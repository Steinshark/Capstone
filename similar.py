import string
import sys
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

f=open(sys.argv[1])

ideal=f.readline().strip()#"UH I GRADUATED FROM UH 2ND GRADE I GOT A LITTLE LEADING FACTOR IN MY SITUATION WAS ONE OF THE UH WORST STUFF IN FIRST PHASE AND UH I DID THREE YEARS OF HIGH SITUATION REALLY THINKING AND EXPERIENCING GAINED ME BEFORE I COULD DO THAT"

real=f.readline().strip()#"NOPE, I WAS WORKING A LOT. MY DAD OWNS A CONTRACTING COMPANY. HE BUILDS HOUSES AND STORES AND STUFF. I GOT A LOT OF EXPERIENCE FROM THAT. I KNEW IF I JOINED AT 18 I WASN’T MATURE ENOUGH YA KNOW. I DON’T THINK I WOULD HAVE MADE IT THEN."


comb=[ideal,real]
vec=CountVectorizer().fit_transform(comb)
vecs=vec.toarray()

sim=cosine_similarity(vecs)

score= sim[0][1]

if(score<0.3):
    print(str(score)+" rare")
else:
    print(str(score)+" common")
