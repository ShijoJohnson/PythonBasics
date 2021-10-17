"""Write a program which will find all such numbers which are divisible by 7 but are not a multiple of 5, between 2000 and 3200 (both included).
The numbers obtained should be printed in a comma-separated sequence on a single line.
"""
lst = []
for num in range(2000,3201):
    if (num % 7 ==0) and (num % 5 !=0) :
        lst.append(str(num))
print(','.join(lst))

l=[]
for i in range(2000, 3201):
    if (i%7==0) and (i%5!=0):
        l.append(str(i))

print(','.join(l))

inp = input("Enter the numbers:")
int_inp = list(map(int,inp.split(',')))
print(int_inp)

l = []
for i in range(1,100):
    if(i%7 == 0 and i%5 != 0):
        l.append(str(i))
print(",".join(l))

### Question 2
"""Level 1

Write a program which can compute the factorial of a given numbers.
The results should be printed in a comma-separated sequence on a single line.
Suppose the following input is supplied to the program:
8
Then, the output should be:
40320"""
def calcFact(n):
    if n ==0:
        return(1)
    else:
        return(n*calcFact(n-1))

print(calcFact(8))

def calcFact():
    inp = input("Enter the number to find factorial: ")
    try:
        inp_int = int(inp)
        if inp_int <0:
            print("Value less than 0")
            return(0)
    except:
        print("Not a valid number")
        return(0)
    product = 1
    for i in range(inp_int):
        product = product * (i+1)
    print(product)
calcFact()


def calcFact(num):
    if num ==0:
        return(1)
    return (num * calcFact(num -1))

print(calcFact(8))


    

### Question 3

"""Question:
With a given integral number n, write a program to generate a dictionary that contains (i, i*i) such that is an integral number 
between 1 and n (both included). and then the program should print the dictionary.
Suppose the following input is supplied to the program:
8
Then, the output should be:
{1: 1, 2: 4, 3: 9, 4: 16, 5: 25, 6: 36, 7: 49, 8: 64}"""


dict = {}
n = 8
for i in range(1,8+1):
    dict[i] = i*i

print(dict)


inp = int(input("Enter the num:"))
mydic = {}
for i in range(1,inp+1):
    mydic[i] = i*i
print(mydic)


"""Hints:
In case of input data being supplied to the question, it should be assumed to be a console input.
Consider use dict()

Solution:
python
n=int(input())
d=dict()
for i in range(1,n+1):
    d[i]=i*i

print(d)
"""




"""
### Question 4
Write a program which accepts a sequence of comma-separated numbers from console and generate a list and a tuple which contains every number.
Suppose the following input is supplied to the program:
34,67,55,33,12,98
Then, the output should be:
['34', '67', '55', '33', '12', '98']
('34', '67', '55', '33', '12', '98')"""
"""
inp = input ("Enter the numbers by comma separated:")
inp_lst = inp.split(',')
print(inp_lst)
print(tuple(inp_lst))

Hints:
In case of input data being supplied to the question, it should be assumed to be a console input.
tuple() method can convert list to tuple

Solution:
```python
values=input()
l=values.split(",")
t=tuple(l)
print(l)
print(t)"""



"""### Question 5
Level 1
Question:
Define a class which has at least two methods:
getString: to get a string from console input
printString: to print the string in upper case.
Also please include simple test function to test the class methods.
"""

class myClass():
    def __init__(self):
        self.inp = ""

    def getStrin(self):
        self.inp = input("Enter String:")
    
    def putString(self):
        print(self.inp.upper())

p = myClass()
p.getStrin()
p.putString()

"""
Solution:
```python
class InputOutString(object):
    def __init__(self):
        self.s = ""

    def getString(self):
        self.s = input()
    
    def printString(self):
        print(self.s.upper())

strObj = InputOutString()
strObj.getString()
strObj.printString()
```"""


"""
### Question 6
Level 2

Question:
Write a program that calculates and prints the value according to the given formula:
Q = Square root of [(2 * C * D)/H]
Following are the fixed values of C and H:
C is 50. H is 30.
D is the variable whose values should be input to your program in a comma-separated sequence.
Example
Let us assume the following comma separated input sequence is given to the program:
100,150,180
The output of the program should be:
18,22,24"""

"""def calc():
    C = 50
    H = 30
    D = input("Enter the numbers by comma:")
    D_lst = map(int, D.split(','))
    out = []

    for i in D_lst:
        out.append(str(round(((2*C*i)/H)**0.5)))
    print(','.join(out))

calc()"""


"""Hints:
If the output received is in decimal form, it should be rounded off to its nearest value (for example, if the output received is 26.0, it should be printed as 26)
In case of input data being supplied to the question, it should be assumed to be a console input. 

Solution:
"""
"""import math
c=50
h=30
value = []
items=[x for x in input().split(',')]
for d in items:
    value.append(str(int(round(math.sqrt(2*c*float(d)/h)))))

print(','.join(value))"""



"""### Question 7
Level 2
Question:
Write a program which takes 2 digits, X,Y as input and generates a 2-dimensional array. 
The element value in the i-th row and j-th column of the array should be i*j.
Note: i=0,1.., X-1; j=0,1,¡­Y-1.
Example
Suppose the following inputs are given to the program:
3,5
Then, the output of the program should be:
[[0, 0, 0, 0, 0], [0, 1, 2, 3, 4], [0, 2, 4, 6, 8]] """

"""
i,j = map(int,input("Enter the numbers separated by comma:").split(','))
print(i, j)

multilist = [[0 for col in range(j)] for row in range(i)]

for row in range(i):
    for col in range(j):
        multilist[row][col] = row*col


print(multilist)"""

"""
### Question 8
Level 2
Question:
Write a program that accepts a comma separated sequence of words as input and prints the words in a comma-separated sequence after sorting them alphabetically.
Suppose the following input is supplied to the program:
without,hello,bag,world
Then, the output should be:
bag,hello,without,world
"""
"""inp = ['a', 'v', 'c']
print(inp)
inp.sort()
print(inp)
inp.sort(reverse = True)
print(inp)
print(sorted(inp))"""


### Question 9
"""Level 2
Question£º
Write a program that accepts sequence of lines as input and prints the lines after making all characters in the sentence capitalized.
Suppose the following input is supplied to the program:
Hello world
Practice makes perfect
Then, the output should be:
HELLO WORLD
PRACTICE MAKES PERFECT

Hints:
In case of input data being supplied to the question, it should be assumed to be a console input.

Solution:

lines = []
while True:
    s = input()
    if s:
        lines.append(s.upper())
    else:
        break;

for sentence in lines:
    print(sentence)
"""

"""lines = ["hello World"]
print(lines)
upper_lines = [c.upper() for c in lines]
print(upper_lines)
"""

### Question 10
"""Level 2
Question:
Write a program that accepts a sequence of whitespace separated words as input and prints the words after removing all duplicate words 
and sorting them alphanumerically.
Suppose the following input is supplied to the program:
hello world and practice makes perfect and hello world again
Then, the output should be:
again and hello makes perfect practice world

Hints:
In case of input data being supplied to the question, it should be assumed to be a console input.
We use set container to remove duplicated data automatically and then use sorted() to sort the data.
"""
"""inp_str = input("Enter strings sep by space: ")
inp = inp_str.split(' ')
inp_set = set(inp)
print(' '.join(sorted(list(inp_set))))"""




### Question 11
"""Level 2

Question:
Write a program which accepts a sequence of comma separated 4 digit binary numbers as its input and then check whether they are divisible by 5 or not. 
The numbers that are divisible by 5 are to be printed in a comma separated sequence.
Example:
0100,0011,1010,1001
Then the output should be:
1010
Notes: Assume the data is input by console.

Solution:
value = []
items=[x for x in input().split(',')]
for p in items:
    intp = int(p, 2)
    if not intp%5:
        value.append(p)

print(','.join(value))
"""
"""inp = input("enter number: ")
inp_lst = inp.split(',')
out_lst = []
for i in inp_lst:
    if int(i,2)%5 == 0:
        print(int(i,2),i)
        out_lst.append(i)
print(','.join(out_lst))
"""



### Question 12
"""Level 2

Question:
Write a program, which will find all such numbers between 1000 and 3000 (both included) such that each digit of the number is an even number.
The numbers obtained should be printed in a comma-separated sequence on a single line.

Solution:"""
"""values = []
for i in range(1000, 3001):
    s = str(i)
    if (int(s[0])%2==0) and (int(s[1])%2==0) and (int(s[2])%2==0) and (int(s[3])%2==0):
        values.append(s)
print(",".join(values))"""
"""
out_val = []
for i in range(1000, 3001):
    stri = str(i)
    if(int(stri[0])%2==0) and (int(stri[1])%2==0) and (int(stri[2])%2==0) and (int(stri[3])%2==0):
        out_val.append(stri)
print(out_val)"""



### Question 13
"""Level 2

Question:
Write a program that accepts a sentence and calculate the number of letters and digits.
Suppose the following input is supplied to the program:
hello world! 123
Then, the output should be:
LETTERS 10
DIGITS 3

Hints:
In case of input data being supplied to the question, it should be assumed to be a console input.

Solution:
s = input()
d={"DIGITS":0, "LETTERS":0}
for c in s:
    if c.isdigit():
        d["DIGITS"]+=1
    elif c.isalpha():
        d["LETTERS"]+=1
    else:
        pass
print("LETTERS", d["LETTERS"])
print("DIGITS", d["DIGITS"])
"""
"""inp = input("Enter the line:")
cnt_Digit = 0
cnt_Alpha = 0

for s in inp:
    if s.isalpha():
        cnt_Alpha+=1
    elif s.isnumeric():
        cnt_Digit+=1
    
print("Letters count =%i" %cnt_Alpha)
print("Digits count =%i" %cnt_Digit)

"""



"""### Question 14
Level 2

Question:
Write a program that accepts a sentence and calculate the number of upper case letters and lower case letters.
Suppose the following input is supplied to the program:
Hello world!
Then, the output should be:
UPPER CASE 1
LOWER CASE 9

Hints:
In case of input data being supplied to the question, it should be assumed to be a console input.

Solution:
s = input()
d={"UPPER CASE":0, "LOWER CASE":0}
for c in s:
    if c.isupper():
        d["UPPER CASE"]+=1
    elif c.islower():
        d["LOWER CASE"]+=1
    else:
        pass
print("UPPER CASE", d["UPPER CASE"])
print("LOWER CASE", d["LOWER CASE"])
"""

"""i = [1,2,3,4,5,6,7,8,9,10,12,14,13]
j = [x for x in i if x%2==0]
print(j)"""


### Question 17
"""Level 2

Question:
Write a program that computes the net amount of a bank account based a transaction log from console input. The transaction log format is shown as following:
D 100
W 200

D means deposit while W means withdrawal.
Suppose the following input is supplied to the program:
D 300
D 300
W 200
D 100
Then, the output should be:
500

Hints:
In case of input data being supplied to the question, it should be assumed to be a console input.
"""
"""netAmount = 0
while True:
    s = input()
    if not s:
        break
    values = s.split(" ")
    operation = values[0]
    amount = int(values[1])
    if operation=="D":
        netAmount+=amount
    elif operation=="W":
        netAmount-=amount
    else:
        pass
print(netAmount)"""

"""netamt = 0
while True:
    str = input()
    if not str:
        break
    values = str.split(" ")
    op = values[0]
    amt = int(values[1])
    if op == "D":
        netamt+=amt
    else:
        netamt-=amt
print(netamt)"""





### Question 18
"""Level 3
Question:
A website requires the users to input username and password to register. Write a program to check the validity of password input by users.
Following are the criteria for checking the password:
1. At least 1 letter between [a-z]
2. At least 1 number between [0-9]
1. At least 1 letter between [A-Z]
3. At least 1 character from [$#@]
4. Minimum length of transaction password: 6
5. Maximum length of transaction password: 12
Your program should accept a sequence of comma separated passwords and will check them according to the above criteria. 
Passwords that match the criteria are to be printed, each separated by a comma.
Example
If the following passwords are given as input to the program:
ABd1234@1,a F1#,2w3E*,2We3345
Then, the output of the program should be:
ABd1234@1

Hints:
In case of input data being supplied to the question, it should be assumed to be a console input.
import re
value = []
items=[x for x in input().split(',')]
for p in items:
    if len(p)<6 or len(p)>12:
        continue
    else:
        pass
    if not re.search("[a-z]",p):
        continue
    elif not re.search("[0-9]",p):
        continue
    elif not re.search("[A-Z]",p):
        continue
    elif not re.search("[$#@]",p):
        continue
    elif re.search("\s",p):
        continue
    else:
        pass
    value.append(p)
print(",".join(value))"""

"""import re
value = []
inp = input("Enter the password: ").split(",")
for p in inp:
    if not re.search("[a-z]",p) :
        continue
    elif not re.search("[0-9]",p) :
        continue
    value.append(p)

print(value)"""

### Question 19
"""Level 3

Question:
You are required to write a program to sort the (name, age, height) tuples by ascending order where name is string, 
age and height are numbers. The tuples are input by console. The sort criteria is:
1: Sort based on name;
2: Then sort based on age;
3: Then sort by score.
The priority is that name > age > score.
If the following tuples are given as input to the program:
Tom,19,80
John,20,90
Jony,17,91
Jony,17,93
Json,21,85
Then, the output of the program should be:
[('John', '20', '90'), ('Jony', '17', '91'), ('Jony', '17', '93'), ('Json', '21', '85'), ('Tom', '19', '80')]

Hints:
In case of input data being supplied to the question, it should be assumed to be a console input.
We use itemgetter to enable multiple sort keys.

Solutions:
from operator import itemgetter, attrgetter

l = []
while True:
    s = input()
    if not s:
        break
    l.append(tuple(s.split(",")))

print(sorted(l, key=itemgetter(0,1,2)))
"""

##first non repeating
"""inp = "abcdefabcef"
for i in inp:
    if inp.find(i) == inp.rfind(i):
        print (i)
        break

y={}
x= {'a':{'b':'c','d':'e'}}
def separate(x):
    for key,val in x.items():
        for k1,v1 in val.items():
            y[key+k1] = v1
    print(y)
separate(x)
{'ab': 'c', 'ad': 'e'}
"""



### Question 21
"""Level 3
Question
A robot moves in a plane starting from the original point (0,0). The robot can move toward UP, DOWN, LEFT and RIGHT with a given steps. 
The trace of robot movement is shown as the following:
UP 5
DOWN 3
LEFT 3
RIGHT 2
¡­
The numbers after the direction are steps. Please write a program to compute the distance from current position after a sequence of movement 
and original point. If the distance is a float, then just print the nearest integer.
Example:
If the following tuples are given as input to the program:
UP 5
DOWN 3
LEFT 3
RIGHT 2
Then, the output of the program should be:
2"""


"""import math
pos=[0,0]
while True:
    inp = input("Values:")
    if not inp:
        break
    dir,steps = inp.split(" ")
    if dir == "UP":
        pos[0]+=int(steps)
    elif dir == "DOWN":
        pos[0]-=int(steps)
    elif dir == "LEFT":
        pos[1]-=int(steps)
    elif dir == "RIGHT":
        pos[1]+=int(steps)

    dist = math.sqrt(pos[0]**2 + pos[1]**2)

print(dist)"""

"""import math
pos = [0,0]
while True:
    s = input()
    if not s:
        break
    movement = s.split(" ")
    direction = movement[0]
    steps = int(movement[1])
    if direction=="UP":
        pos[0]+=steps
    elif direction=="DOWN":
        pos[0]-=steps
    elif direction=="LEFT":
        pos[1]-=steps
    elif direction=="RIGHT":
        pos[1]+=steps
    else:
        pass
print(int(round(math.sqrt(pos[1]**2+pos[0]**2))))"""



"""inp = input("Enter lines:")
lst = list(inp.split(" "))
dic = {}

for word in lst:
    dic[word] = dic.get(word,0) + 1

print(sorted(dic.items()))"""



### Question 22
"""Level 3

Question:
Write a program to compute the frequency of the words from the input. The output should output after sorting the key alphanumerically. 
Suppose the following input is supplied to the program:
New to Python or choosing between Python 2 and Python 3? Read Python 2 or Python 3.
Then, the output should be:
2:2
3.:1
3?:1
New:1
Python:5
Read:1
and:1
between:1
choosing:1
or:2
to:1

Hints
In case of input data being supplied to the question, it should be assumed to be a console input.

Solution:

freq = {}   # frequency of words in text
line = input()
for word in line.split():
    freq[word] = freq.get(word,0)+1

words = freq.keys()
words.sort()

for w in words:
    print("%s:%d" % (w,freq[w]))"""



### Question 33
#Define a function which can print a dictionary where the keys are numbers between 1 and 3 (both included) and the values are square of keys.

"""dict = {}
for i in range(4):
    dict[i] = i*i
print (dict)
print(dict.items())
print(dict.keys())
print(dict.values())
"""
"""
li = [1,2,3,4,5,6,7,8,9,10]
print([x for x in li if x%2 ==0])
print(list(filter(lambda x: x%2 ==0, li)))"""

"""li = [1,2,3,4,5,6,7,8,9,10]
squaredNumbers = list(map(lambda x: x ** 2, li))
print(squaredNumbers)
"""

"""li = [1,2,3,4,5,6,7,8,9,10]
print(list(map(lambda x: x**2,list(filter(lambda x: x%2==0, li)))))
"""

"""Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Output: Because nums[0] + nums[1] == 9, we return [0, 1].
"""
"""nums = [2,7,11,15]
target = 9
print("Hello")
def twoSum(nums, target):
    d = {}
    for i, n in enumerate(nums):    1,7
        m = target - n             2
        if m in d:
            print([d[m], i])
        else:
            d[n] = i
    print(d)
twoSum(nums,target)
"""
"""d = {0:10,1:20,3:30}
for i in d:
    print(i, d[i])
0 10
1 20
3 30"""


"""Given a string s, find the length of the longest substring without repeating characters.
Input: s = "abcabcbb"
Output: 3
Explanation: The answer is "abc", with the length of 3.
"""


"""used = {}
max_length = start = 0
for i, c in enumerate(s):
    if c in used and start <= used[c]:
        start = used[c] + 1
    else:
        max_length = max(max_length, i - start + 1)
        
    used[c] = i
return max_length"""


"""def longestPalindrome(s):
    res = ""
    for i in range(len(s)):        
        odd  = palindromeAt(s, i, i)
        even = palindromeAt(s, i, i+1)
        
        res = max(res, odd, even, key=len)
    return res
 
# starting at l,r expand outwards to find the biggest palindrome
def palindromeAt( s, l, r):    
    while l >= 0 and r < len(s) and s[l] == s[r]:
        l -= 1
        r += 1
    return s[l+1:r]

print(longestPalindrome("babad"))
print(longestPalindrome("cbbd"))"""


"""The string "PAYPALISHIRING" is written in a zigzag pattern on a given number of rows like this: (you may want to display this pattern in a fixed font for better legibility)

P   A   H   N
A P L S I I G
Y   I   R

And then read line by line: "PAHNAPLSIIGYIR"

Write the code that will take a string and make this conversion given a number of rows:

string convert(string s, int numRows);

class Solution(object):
    def convert(self, s, numRows):
        '''
        PAYPALISHIRING
        3 rows
        
        P   A
        A P L H
        Y   I
        
        PAHNAPLSIIGYIR
        
        1. Approach 1
        
        0 [PA]  -> [P,A,]
        1 [APL]
        2 [Y]
        
        Time O(n)
        Space O(n)
        
        n is the number of characters in string
        k is the numRows
        '''
        
        if numRows == 1 or numRows >= len(s):
            return s
        
        delta = -1
        row = 0
        res = [[] for i in xrange(numRows)]
        
        #iterate through our string
        for c in s:
            res[row].append(c)
            if row == 0 or row == numRows - 1:
                delta *= -1
            row += delta
        
        #consolidate result
        for i in xrange(len(res)):
            res[i] = ''.join(res[i])
        return ''.join(res)"""


"""11. Container With Most Water
Medium

Given n non-negative integers a1, a2, ..., an , where each represents a point at coordinate (i, ai). n vertical lines are drawn such that the two endpoints of the line i is at (i, ai) and (i, 0). Find two lines, which, together with the x-axis forms a container, such that the container contains the most water.

Notice that you may not slant the container.

Example 1:

Input: height = [1,8,6,2,5,4,8,3,7]
Output: 49
class Solution:
    def maxArea(self, height):
        i, j = 0, len(height) - 1
        water = 0
        while i < j:
            water = max(water, (j - i) * min(height[i], height[j]))
            if height[i] < height[j]:
                i += 1
            else:
                j -= 1
        return water"""

"""Write a function to find the longest common prefix string amongst an array of strings.

If there is no common prefix, return an empty string "".

Example 1:

Input: strs = ["flower","flow","flight"]
Output: "fl"

 def longestCommonPrefix(self, strs):

        if not strs:
            return ""
        shortest = min(strs,key=len)
        for i, ch in enumerate(shortest):
            for other in strs:
                if other[i] != ch:
                    return shortest[:i]
        return shortest """

"""def longestCommonPrefix(my_str):

    if not my_str:
        return ''
    prefix = my_str[0]
    for word in my_str:
        if len(prefix) > len(word):
            prefix, word = word, prefix
            
        while len(prefix) > 0:
            if word[:len(prefix)] == prefix:
                break
            else:
                prefix = prefix[:-1]
    return prefix     
    
my_list = ['car', 'carbon', 'carbonmonoxide']  
result = longestCommonPrefix(my_list)
print(result)"""



"""Suppose we have an array of numbers. It stores n integers, there are there elements a, b, c in the array, such that a + b + c = 0. 
Find all unique triplets in the array which satisfies the situation. So if the array is like [-1,0,1,2,-1,-4], 
then the result will be [[-1, 1, 0], [-1, -1, 2]]
class Solution(object):
   def threeSum(self, nums):
      nums.sort()           [-4,-1,-1,0,1,2]   ********************************************
      result = []
      for i in range(len(nums)-2):
         if i> 0 and nums[i] == nums[i-1]:
            continue
         l = i+1
         r = len(nums)-1
         while(l<r):
            sum = nums[i] + nums[l] + nums[r]
            if sum<0:
               l+=1
            elif sum >0:
               r-=1
            else:
               result.append([nums[i],nums[l],nums[r]])
               while l<len(nums)-1 and nums[l] == nums[l + 1] : l += 1
               while r>0 and nums[r] == nums[r - 1]: r -= 1
               l+=1
               r-=1
      return result
ob1 = Solution()
print(ob1.threeSum([-1,0,1,2,-1,-4]))"""



"""Given a string containing digits from 2-9 inclusive, return all possible letter combinations that the number could represent. Return the answer in any order.

A mapping of digit to letters (just like on the telephone buttons) is given below. Note that 1 does not map to any letters.
Input: digits = "23"
Output: ["ad","ae","af","bd","be","bf","cd","ce","cf"]

def letterCombinations(digits):
    interpret_digit = {
        '1': '',
        '2': 'abc',
        '3': 'def',
        '4': 'ghi',
        '5': 'jkl',
        '6': 'mno',
        '7': 'pqrs',
        '8': 'tuv',
        '9': 'wxyz',
        '0': ' '}
    all_combinations = [''] if digits else []
    for digit in digits:
        current_combinations = list()
        for letter in interpret_digit[digit]:
            for combination in all_combinations:   **********************************************************
                current_combinations.append(combination + letter)
        all_combinations = current_combinations
    return all_combinations
digits = "2349"
print(letterCombinations(digits))
"""



"""
Given the head of a linked list, remove the nth node from the end of the list and return its head.
def removeNthFromEnd(head: ListNode, n):
    fast = slow = head                ******************************************
    for _ in range(n):
        fast = fast.next
    if not fast:
        return head.next
    while fast.next:
        fast = fast.next
        slow = slow.next
    slow.next = slow.next.next
    return head

head = [1,2,3,4,5]
n = 2    
print(removeNthFromEnd(head, n))"""



"""Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.
Example 1:
Input: n = 3
Output: ["((()))","(()())","(())()","()(())","()()()"]
class Solution(object):
    def generateParenthesis(self, N):
        ans = []
        def backtrack(S = '', left = 0, right = 0):
            if len(S) == 2 * N:
                ans.append(S)
                return
            if left < N:
                backtrack(S+'(', left+1, right)     ******************************************
            if right < left:
                backtrack(S+')', left, right+1)     ******************************************

        backtrack()
        return ans"""


"""24. Swap Nodes in Pairs
Given a linked list, swap every two adjacent nodes and return its head.
class Solution:
    def swapPairs(self, head):
        dummy = pre = ListNode(0)
        pre.next = head
        while pre.next and pre.next.next:
            a = pre.next
            b = a.next
            pre.next, a.next, b.next = b, b.next, a
            pre = a
        return dummy.next
"""

"""Given a sorted array nums, remove the duplicates in-place such that each element appears only once and returns the new length.
Do not allocate extra space for another array, you must do this by modifying the input array in-place with O(1) extra memory.
Input: nums = [0,0,1,1,1,2,2,3,3,4]
Output: 5, nums = [0,1,2,3,4]
"""
"""a = [1,1,1,3,4,5,5,5,5,7,7,7,9,9]
for i in range(len(a)-1,0,-1):
    if a[i] == a[i-1]:                         
        del a[i]                            **************************************
print(len(a))
print(a)"""

"""print(2%9)
print(9%2)"""


"""Given an array nums of integers, return how many of them contain an even number of digits.
Input: nums = [12,345,2,6,7896]
Output: 2

    def findNumbers(self, nums: List[int]) -> int:
        return sum((len(str(i))) % 2 == 0 for i in nums)"""

"""1290. Convert Binary Number in a Linked List to Integer
    def getDecimalValue(self, head: ListNode) -> int:
        answer = 0
        while head: 
            answer = 2*answer + head.val 
            head = head.next 
        return answer """        

"""
1281. Subtract the Product and Sum of Digits of an Integer
import operator, functools
A = list(map(int,'234'))
print(functools.reduce(operator.mul,A))   
print(sum(A))"""


"""1275. Find Winner on a Tic Tac Toe Game
def tictactoe(self, moves: List[List[int]]) -> str:
        n = 3
        rows, cols = [0] * n, [0] * n
        diag1 = diag2 = 0
        for index, move in enumerate(moves):
            i, j = move
            sign = 1 if index % 2 == 0 else -1  ****************************************
            rows[i] += sign
            cols[j] += sign
            if i == j:
                diag1 += sign
            if i + j == n-1:
                diag2 += sign
            if abs(rows[i]) == n or abs(cols[j]) == n or abs(diag1) == n or abs(diag2) == n:
                return 'A' if sign == 1 else 'B'
        return "Draw" if len(moves) == (n * n) else 'Pending'"""

"""1189. Maximum Number of Balloons
Example 2:

Input: text = "loonbalxballpoon"
Output: 2

    def maxNumberOfBalloons(self, text: str) -> int:
        counter = {"b":0, "a":0, "l":0, "o":0, "n":0}
        for char in text:
            if char in counter:
                counter[char] += 1
        counter["l"] //= 2
        counter["o"] //= 2
        return min(counter.values())"""                  """"***************************************"""

##Git Integration