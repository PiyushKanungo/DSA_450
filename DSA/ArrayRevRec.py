def reverseList(A,start,end):
    if start>=end:
        return
    A[start],A[end] = A[end], A[start]
    reverseList( A, start+1, end-1)

#Driver Function
A = [1,2,3,4,5,6]
print(A)
reverseList(A,0,5)
print("reversed List is")
print(A)


