""" 
Lab-90 : Regular Expression
1. Quick search of substring in a string as a pattern 
"""
#%%
# dependencies for the regular expression
import re

match1 = re.search('Ajeet', 'Ajeet Kumar is exceptional teacher in applied mathematics!')
match2 = re.search('applied mathematics', 'Ajeet Kumar is exceptional teacher in applied mathematics!')
print(match1)
print(match2)


# %%
# function for matching regular expression patterns
def find_pattern(text, patterns):
    return re.search(patterns, text)

text = "Hi, How are you Ajeet ?"
pattern = "Ajeet"
print(find_pattern(text, pattern))

#%%
## Quantifiers
# zero or more
print(find_pattern("ac", "ab*"))
print(find_pattern("abc", "ab*"))
print(find_pattern("abbc", "ab*"))
# check wheather pattern is present or absent
print(find_pattern("ac", "ab?"))
print(find_pattern("abc", "ab?"))
print(find_pattern("abbc", "ab?"))
# one or more
print(find_pattern("ac", "ab+"))
print(find_pattern("abc", "ab+"))
print(find_pattern("abbc", "ab+"))
# {n}: Matches if a character is present exactly n number of times
print(find_pattern("abbc", "ab{2}"))
# {m,n}: Matches if a character is present from m to n number of times
print(find_pattern("aabbbbbbc", "ab{3,5}"))   # return true if 'b' is present 3-5 times
print(find_pattern("aabbbbbbc", "ab{7,10}"))  # return true if 'b' is present 7-10 times
print(find_pattern("aabbbbbbc", "ab{,10}"))   # return true if 'b' is present atmost 10 times
print(find_pattern("aabbbbbbc", "ab{10,}"))   # return true if 'b' is present from at least 10 times

# %%
## Anchors
# '^':  start of a string
# '$': end of string

print(find_pattern("Ajeet", "^A")) #return true if string starts with 'A'
print(find_pattern('Indian', 'n$')) # return true if string ends with 'n'


# %%
# Wildcard
# '.' : matches any character
print(find_pattern("a", "."))
print(find_pattern("%", "." ))

# %%
"""
Character sets : 
We will search for '[' and ']' because they are used for
specifying a character class, which is set of characters that


"""
# Characters can be listed individually as follows
print(find_pattern("a", "[abc]"))

# a range of characters can be indicated by giving two characters and separating them by a '-'.
print(find_pattern("c", "[a-c]"))

# '^' is used inside character set to indicate complementary set
print(find_pattern("a", "[^abc]"))  # return true if neither of these is present



# %%
# Greedy vs Non-greedy regex
print(find_pattern("aabbbbbb", "ab{3,5}")) # # return if a is followed by b 3-5 times GREEDY
print(find_pattern("aabbbbbb", "ab{3,5}?")) # return if a is followed by b 3-5 times GREEDY
# Example of HTML code
print(re.search("<.*>","<HTML><TITLE>My Page</TITLE></HTML>"))
# Example of HTML code
print(re.search("<.*?>","<HTML><TITLE>My Page</TITLE></HTML>"))


# %%
""" 
match() Determine if the RE matches at the beginning of the string

search() Scan through a string, looking for any location where this RE matches

finall() Find all the substrings where the RE matches, and return them as a list

finditer() Find all substrings where RE matches and return them as asn iterator

sub() Find all substrings where the RE matches and substitute them with the given string
"""

def match_pattern(text, patterns):
    if re.match(patterns, text):
        return re.match(patterns, text)
    else: 
        return "Not Found!"
    

print(find_pattern("abbc", "b+"))
print(match_pattern("abbc", "b+"))


## Example usage of the sub() function. Replace Road with rd.

street = '21 Ramakrishna Road'
print(re.sub('Road', 'Rd', street))

print(re.sub('R\w+', 'Rd', street))

## Example usage of finditer(). Find all occurrences of word Festival in given sentence

text = 'Diwali is a festival of lights, Holi is a festival of colors!'
pattern = 'festival'
for match in re.finditer(pattern, text):
    print('START -', match.start(), end="")
    print('END -', match.end())

# Example usage of findall(). In the given URL find all dates
url = "http://www.telegraph.co.uk/formula-1/2017/10/28/mexican-grand-prix-2017-time-does-start-tv-channel-odds-lewisl/2017/05/12"
date_regex = '/(\d{4})/(\d{1,2})/(\d{1,2})/'
print(re.findall(date_regex, url))

# exploring groupds
m1 = re.search(date_regex, url)
print(m1.group())

print(m1.group(1))
print(m1.group(2))
print(m1.group(3))
print(m1.group(0))

# %%
"""
Syntactic Processing

"""
import nltk
