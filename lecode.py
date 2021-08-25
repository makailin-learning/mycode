#题解：不单独去判断正负，也不单独去判断是否溢出
def reverse1(x):
    ret=0
    temp=0
    last=0
    while(x!=0):
        temp=x%10
        last=ret
        ret=ret*10+temp
        #未溢出时，会相等
        if last!=ret//10:  #这里是双斜杠，python里的除法有两种，//对应C语言中的/，而/对应C语言中的float型除法
            return 0
        x=x//10
    return ret
