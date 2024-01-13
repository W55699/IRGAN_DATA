# 读取txt文件内容
with open("test.txt", "r") as file:
    input_numbers = file.read()

# 将数字字符串按行拆分
lines = input_numbers.split('\n')

# 将每行数字转为两列
output_lines = []
for line in lines:
    numbers_list = line.split()
    output_lines.extend([f"{numbers_list[0]} {number}" for number in numbers_list[1:]])

# 将结果写入txt文件
with open("test1.txt", "w") as file:
    file.write("\n".join(output_lines))

print("转换完成")
