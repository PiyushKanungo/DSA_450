def reverseString(s,start,end):
        while start<end:
            s[start],s[end]=s[end],s[start]
            start+=1
            end-=1
   
 
# Driver function to test above function
A = [1, 2, 3, 4, 5, 6]
print(A)
reverseString(A, 0, 5)
print("Reversed list is")
print(A)