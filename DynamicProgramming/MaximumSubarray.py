'''
给定一个整数数组 nums ，找到一个具有最大和的连续子数组
（子数组最少包含一个元素），返回其最大和。

示例:

输入: [-2,1,-3,4,-1,2,1,-5,4],
输出: 6
解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。


'''

class Solution:
    def maxSubArray(self, nums) -> int:
        if len(nums) == 0:
            return 0
        bestNum = {
            0: nums[0]
        }
        sum = nums[0]
        for inum in range(1, len(nums)):
            bestNum[inum] = max(
                nums[inum],
                bestNum[inum-1] + nums[inum]
            )
            sum = max(
                sum, bestNum[inum]
            )
        return sum

    def divideMaxSubArray(self, nums) -> int:
        if len(nums) <= 0:
            return 0
        if len(nums) == 1:
            return nums[0]
        else:
            partition = len(nums) // 2

            left = self.divideMaxSubArray(nums[0: partition])
            right = self.divideMaxSubArray(nums[partition:])

            leftMax = 0
            leftSum = 0
            for k in range(partition-1, -1, -1):
                leftSum += nums[k]
                leftMax = leftSum if leftSum > leftMax else leftMax

            rightMax = 0
            rightSum = 0
            for k in range(partition, len(nums)):
                rightSum += nums[k]
                rightMax = rightSum if rightSum > rightMax else rightMax


            return max(max(left, right), leftMax+rightMax)


if __name__ == '__main__':
    s = Solution()
    data = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    print(s.maxSubArray([-2,1,-3,4,-1,2,1,-5,4]))
    print(s.divideMaxSubArray([-2,1,-3,4,-1,2,1,-5,4]))
