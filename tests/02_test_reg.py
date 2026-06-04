import re

md_content = "这里有一张图 ![示意图](images/demo(1).png) 再来一张 ![图片](images/demo(1).png)"
image_name = "demo(1).png" # -> "demo\(1\).png"

print(re.escape(image_name))

# 编译正则（不变） raw  \!  \[ 转义  ! 非  [ 5:10 ] -> 特殊的含义
#  . 任意字符串
#  *  数量 0 - n + 1-n
#  ?  贪婪匹配  非贪婪   ![示意图]
# 不加问号是贪婪   ![示意图](images/demo(1).png) 再来一张 ![图片]
rep = re.compile(r"\!\[.*?\]\(.*?" + re.escape(image_name) + r".*?\)")

# 1. search：全文找【第一个】匹配 → 返回 match 对象
# 匹配第一个
# Match group() -> 匹配的内容 ![示意图](images/demo(1).png)  .start() 起始坐标  .end() 结束坐标  ==  span() [0] [1]
match_obj = rep.search(md_content)
print("1. search 结果：", match_obj.group() if match_obj else "未找到")
print("1. search 结果 start：", match_obj.start())
print("1. search 结果 end：", match_obj.end())
print("1. search 结果 start和end：", match_obj.span()[0] , match_obj.span()[1])
#
# # 2. match：只从【开头】匹配 → 这里开头不是图片语法，必找不到
# 相同点: 返回值都是Match匹配结果封装对象,不是具体的结果
# 不同点: 从头匹配
match_first = rep.match(md_content)
print("2. match 结果：", "找到" if match_first else "未找到（只匹配开头）")

# # 3. findall：找【所有】匹配 → 返回字符串列表
# 所有匹配内容
all_list = rep.findall(md_content)
print("3. findall 列表：", all_list)
#
# # 4. finditer：找【所有】匹配 → 返回 match 对象（你要的“search 匹配所有”）
# 返回所有的匹配项 Match
print("4. finditer 遍历所有：")
for m in rep.finditer(md_content):
    print("  - 内容：", m.group(), " | 位置：", m.span())

# # () 和 ?  贪婪匹配
text = "A123B456B"
# # 1. 用 .*  → 贪婪（吃到最后）
print(re.findall(r"A(.*)B", text))  # 输出： 123B456
# # 2. 用 .*? → 非贪婪（找到第一个就停）
print(re.findall(r"A(.*?)B", text)) # 输出： 123

re.sub()
