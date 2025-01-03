

def meteor(generated:str,gold:str)->float:
    m=0
    generated_grams=set(generated.split(" "))
    for token in gold.split(" "):
        if token in generated_grams:
            m+=1
            
    
    
    r=m/len(generated.split(" "))
    p=m/len(gold.split(" "))
    print(f"p: {p}, r: {r}, m: {m}")
    f_mean=(10*p*r)/(r+9*p)
    c=chunk_search(generated,gold)
    print(f"chunks: {c}")
    penalty= 0.5*(c/m)**3
    print(f"penalty: {penalty}")
  
    return f_mean*(1-penalty)
    
    
def chunk_search(s1,s2)->int:
    found_chunks=[]
    s1=s1.split(" ")
    s2=s2.split(" ")
    start_len=len(s1)
    if len(s1)>len(s2):
        start_len=len(s2)
    while start_len>0:
        s1_grams=windows(s1,start_len)
        s2_grams=windows(s2,start_len)
        start=0
        for gram in s1_grams:
            if gram in s2_grams:
                index_set={x for x in range(start,start+len(gram))}
                add_flag=True
                for found_chunk in found_chunks:
                    if index_set.issubset(found_chunk):
                        add_flag=False
                        break
                if add_flag:
                    found_chunks.append(index_set)
            start+=1
        start_len-=1
    
    return len(found_chunks)
        
def windows(s:list[str],w:int)->list[list[str]]:
    grams=[]
    for i in range(len(s)):
         if i+w>len(s):
             break
         grams.append(s[i:i+w])
    return grams 
         
    
print(meteor('the cat sat on the mat','on the mat sat the cat'))


