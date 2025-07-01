#include <bits/stdc++.h>

using namespace std;

// DataClass
class Node
{
public:
    // which column to ask about
    int criteriaAttributeIndex;
    // parent attribute Value
    string attributeValue;
    int treeIndex;
    bool isLeaf;
    string label;
    vector<int> children;
    // numerical features
    bool isNumerical;
    double threshold;
    Node()
    {
        isLeaf = false;
        isNumerical = false;
        threshold = 0.0;
    }
};

// Intermediate helping class
class Table
{
public:
    vector<string> attributeName;
    vector<vector<string>> data;
    // store the unique values in each column
    vector<vector<string>> attributeValueList;
    vector<bool> isNumericalAttribute;
    //Use reference instead of copying
    vector<int> rowIndices;
    Table *originalTable;
    //helpers for getting light copy
    Table(Table *original = nullptr) : originalTable(original) {}
    vector<string> &getRowData(int i)
    {
        if (originalTable != nullptr && !rowIndices.empty())
        {
            return originalTable->data[rowIndices[i]];
        }
        else
        {
            return data[i];
        }
    }
    int getDataSize()
    {
        if (originalTable != nullptr && !rowIndices.empty())
        {
            return rowIndices.size();
        }
        else
        {
            return data.size();
        }
    }
    // helper for numerical column check
    bool checknumerical(int colIndex)
    {
        if (attributeName[colIndex] == "Id")
            return false;
        if (colIndex == attributeName.size() - 1)
            return false;
        for (int i = 0; i < min(10, (int)data.size()); i++)
        {
            try
            {
                stod(data[i][colIndex]); // if numerical the conversion is successful
            }
            catch (...)
            {
                return false;
            }
        }
        return true;
    }

    void extractAttributeValue()
    {
        // make space for each column
        attributeValueList.resize(attributeName.size());
        isNumericalAttribute.resize(attributeName.size());
        // column
        for (int j = 0; j < attributeName.size(); j++)
        {
            isNumericalAttribute[j] = checknumerical(j);
            // unique attributes marked as 1
            if (!isNumericalAttribute[j])
            {
                map<string, int> value;
                // row
                for (int i = 0; i < data.size(); i++)
                {
                    value[data[i][j]] = 1;
                }
                //--//
                for (auto iter = value.begin(); iter != value.end(); iter++)
                {
                    attributeValueList[j].push_back(iter->first);
                }
            }
        }
    }
};
// Split into training and testing data
class DataSplitter
{
public:
    static pair<Table, Table> splitData(Table originalTable, double trainRatio = 0.8)
    {
        Table trainTable, testTable;
        // copy attr names in both the tables
        trainTable.attributeName = originalTable.attributeName;
        testTable.attributeName = originalTable.attributeName;
        vector<int> indices;
        for (int i = 0; i < originalTable.getDataSize(); i++)
        {
            indices.push_back(i);
        }
        random_device rd;
        mt19937 gen(rd());
        shuffle(indices.begin(), indices.end(), gen);

        int trainSize = (int)(originalTable.getDataSize() * trainRatio);
        for (int i = 0; i < indices.size(); i++)
        {
            if (i < trainSize)
            {
                trainTable.data.push_back(originalTable.data[indices[i]]);
            }
            else
            {
                testTable.data.push_back(originalTable.data[indices[i]]);
            }
        }
        return make_pair(trainTable, testTable);
    }
};

// decision tree learning algorithm
class DecisionTree
{
public:
    Table initialTable;
    vector<Node> tree;
    typedef double (DecisionTree::*GainFunction)(Table, int);
    GainFunction currentGainFunction;
    int maxDepth;
    void setGainMethod(string method)
    {
        if (method == "IG")
        {
            currentGainFunction = &DecisionTree::getInfoGain;
            cout << "Using Information Gain (IG)" << endl;
        }
        else if (method == "IGR")
        {
            currentGainFunction = &DecisionTree::getInfoGainRatio;
            cout << "Using Information Gain Ratio (IGR)" << endl;
        }
        else if (method == "NWIG")
        {
            currentGainFunction = &DecisionTree::getNormalizedWeightedInfoGain;
            cout << "Using Normalized Weighted Information Gain (NWIG)" << endl;
        }
        else
        {
            currentGainFunction = &DecisionTree::getInfoGain;
            cout << "Unknown method '" << method << "'. Using default Information Gain (IG)" << endl;
        }
    }

    DecisionTree(Table table, string gainMethod = "IG", int maxDepth = 0)
    {
        initialTable = table;
        initialTable.extractAttributeValue();\
        //added for choosing gain method when running
        setGainMethod(gainMethod);
        //added for implementing maxDepth
        this->maxDepth = maxDepth;
        maxDepth == 0 ? cout << "MaxDepth:Unlimited" : cout << "MaxDepth: " << maxDepth << endl;
        Node root;
        root.treeIndex = 0;
        tree.push_back(root);
        run(initialTable, 0, 0);
    }

    void run(Table table, int nodeIndex, int currentDepth)
    {
        // checking for max depth
        if (maxDepth > 0 && currentDepth >= maxDepth)
        {
            tree[nodeIndex].isLeaf = true;
            tree[nodeIndex].label = getMajorityLabel(table).first;
            return;
        }
        // are the leaves sorted correctly
        if (isLeafNode(table) == true)
        {
            tree[nodeIndex].isLeaf = true;
            tree[nodeIndex].label = table.getRowData(0).back();
            return;
        }

        auto bestSplit = getBestSplit(table);
        int selectedAttributeIndex = bestSplit.first;

        if (selectedAttributeIndex == -1)
        {
            tree[nodeIndex].isLeaf = true;
            tree[nodeIndex].label = getMajorityLabel(table).first;
            return;
        }
        tree[nodeIndex].criteriaAttributeIndex = selectedAttributeIndex;
        tree[nodeIndex].isNumerical = initialTable.isNumericalAttribute[selectedAttributeIndex];
        if (tree[nodeIndex].isNumerical)
        {
            // Numerical splitting
            tree[nodeIndex].threshold = bestSplit.second;

            Table leftTable(&initialTable), rightTable(&initialTable);
            leftTable.attributeName = table.attributeName;
            rightTable.attributeName = table.attributeName;
            for (int i = 0; i < table.getDataSize(); i++)
            {
                //implementing reference copy only not the actual data
                // double value = stod(table.data[i][selectedAttributeIndex]);
                vector<string> &row = table.getRowData(i);
                double value = stod(row[selectedAttributeIndex]);
                if (value <= tree[nodeIndex].threshold)
                {
                    if (table.originalTable != nullptr)
                    {
                        leftTable.rowIndices.push_back(table.rowIndices[i]);
                    }
                    else
                    {
                        leftTable.rowIndices.push_back(i);
                    }
                }
                else
                {
                    if (table.originalTable != nullptr)
                    {
                        rightTable.rowIndices.push_back(table.rowIndices[i]);
                    }
                    else
                    {
                        rightTable.rowIndices.push_back(i);
                    }
                }
            }

            // Create two children
            Node leftChild, rightChild;
            leftChild.treeIndex = tree.size();
            tree[nodeIndex].children.push_back(leftChild.treeIndex);
            tree.push_back(leftChild);

            rightChild.treeIndex = tree.size();
            tree[nodeIndex].children.push_back(rightChild.treeIndex);
            tree.push_back(rightChild);

            if (leftTable.getDataSize() > 0)
            {
                run(leftTable, leftChild.treeIndex, currentDepth + 1);
            }
            else
            {
                tree[leftChild.treeIndex].isLeaf = true;
                tree[leftChild.treeIndex].label = getMajorityLabel(table).first;
            }

            if (rightTable.getDataSize() > 0)
            {
                run(rightTable, rightChild.treeIndex, currentDepth + 1);
            }
            else
            {
                tree[rightChild.treeIndex].isLeaf = true;
                tree[rightChild.treeIndex].label = getMajorityLabel(table).first;
            }
        }
        else
        {

            // best attribute to ask the question on
            // int selectedAttributeIndex = getSelectedAttribute(table);

            // Which rows contain this attributeValue
            map<string, vector<int>> attributeValueMap;
            for (int i = 0; i < table.getDataSize(); i++)
            {
                vector<string> &row = table.getRowData(i);
                attributeValueMap[row[selectedAttributeIndex]].push_back(i);
                // attributeValueMap[table.data[i][selectedAttributeIndex]].push_back(i);
            }
            // save the question
            tree[nodeIndex].criteriaAttributeIndex = selectedAttributeIndex;

            // Check if one of the label is overwhelmingly more common(Majority Voting)
            pair<string, int> majority = getMajorityLabel(table);
            /*if ((double)majority.second / table.data.size() > 0.8)
            {
                tree[nodeIndex].isLeaf = true;
                tree[nodeIndex].label = majority.first;
                return;
            }*/

            // Branch creation through recursion
            for (int i = 0; i < initialTable.attributeValueList[selectedAttributeIndex].size(); i++)
            {
                string attributeValue = initialTable.attributeValueList[selectedAttributeIndex][i];

                // Creating subtable for branch
                // branch means to create table for each attribute values under this attribute
                Table nextTable;
                nextTable.attributeName = table.attributeName;
                vector<int> candidate = attributeValueMap[attributeValue];
                for (int j = 0; j < candidate.size(); j++)
                {
                    nextTable.data.push_back(table.getRowData(candidate[j]));
                }

                // Create new node for this branch
                Node nextNode;
                nextNode.attributeValue = attributeValue;
                nextNode.treeIndex = (int)tree.size();
                tree[nodeIndex].children.push_back(nextNode.treeIndex);
                tree.push_back(nextNode);

                // Handle empty data cases
                if (nextTable.getDataSize() == 0)
                {
                    nextNode.isLeaf = true;
                    nextNode.label = getMajorityLabel(table).first;
                    tree[nextNode.treeIndex] = nextNode;
                }
                else
                {
                    run(nextTable, nextNode.treeIndex, currentDepth + 1);
                }
            }
        }
    }

    // For all datas
    double getEntropy(Table table)
    {
        double ret = 0.0;

        int itemCount = (int)table.getDataSize();
        // Keeping the count of final answer
        map<string, int> labelCount;
        for (int i = 0; i < table.getDataSize(); i++)
        {
            vector<string> &row = table.getRowData(i);
            labelCount[row.back()]++;
            // labelCount[table.data[i].back()]++;
        }
        // calculation of entropy
        for (auto iter = labelCount.begin(); iter != labelCount.end(); iter++)
        {
            double p = (double)iter->second / itemCount;

            ret += -1.0 * p * log(p) / log(2);
        }
        return ret;
    }

    // For all the attribute values under a certain attribute index find the entropy
    double getEntropyAttr(Table table, int attributeIndex)
    {
        double ret = 0.0;
        int itemCount = (int)table.getDataSize();

        // Get all the attribute values under a certain attribute index
        map<string, vector<int>> attributeValueMap;
        for (int i = 0; i < table.getDataSize(); i++)
        {
            vector<string> &row = table.getRowData(i);
            attributeValueMap[row[attributeIndex]].push_back(i);
            // attributeValueMap[table.data[i][attributeIndex]].push_back(i);
        }

        // Temporarily make the branch table for calculation purposes for each of the attribute values
        for (auto iter = attributeValueMap.begin(); iter != attributeValueMap.end(); iter++)
        {
            Table nextTable;
            for (int i = 0; i < iter->second.size(); i++)
            {
                nextTable.data.push_back(table.getRowData(iter->second[i]));
            }
            int nextItemCount = (int)nextTable.getDataSize();
            ret += (double)nextItemCount / itemCount * getEntropy(nextTable);
        }
        return ret;
    }
    // IG
    double getInfoGain(Table table, int attributeIndex)
    {
        return getEntropy(table) - getEntropyAttr(table, attributeIndex);
    }
    // check the test data against the decision tree
    string guess(vector<string> row)
    {
        string label = "";
        int leafNode = dfs(row, 0);
        if (leafNode == -1)
        {
            return "dfs failed";
        }
        label = tree[leafNode].label;
        return label;
    }
    // helper for the test guess
    int dfs(vector<string> &row, int here)
    {
        if (tree[here].isLeaf)
        {
            return here;
        }

        int criteriaAttributeIndex = tree[here].criteriaAttributeIndex;

        if (tree[here].isNumerical)
        {
            // Handle numerical attributes
            double value = stod(row[criteriaAttributeIndex]);
            if (value <= tree[here].threshold)
            {
                // Go to left child (first child)
                if (tree[here].children.size() > 0)
                {
                    return dfs(row, tree[here].children[0]);
                }
            }
            else
            {
                // Go to right child (second child)
                if (tree[here].children.size() > 1)
                {
                    return dfs(row, tree[here].children[1]);
                }
            }
        }
        else
        {
            // Handle categorical attributes
            for (int i = 0; i < tree[here].children.size(); i++)
            {
                int next = tree[here].children[i];
                if (row[criteriaAttributeIndex] == tree[next].attributeValue)
                {
                    return dfs(row, next);
                }
            }
        }

        return -1; // No matching path found
    }
    // helper to check if the leaves all have the same label
    bool isLeafNode(Table table)
    {
        for (int i = 1; i < table.getDataSize(); i++)
        {
            /*if (table.data[0].back() != table.data[i].back())
            {
                return false;
            }*/
            vector<string> &row1 = table.getRowData(0);
            vector<string> &row2 = table.getRowData(i);
            if (row1.back() != row2.back())
            {
                return false;
            }
        }
        return true;
    }

    // trying with just info gain at first
    pair<int, double> getBestSplit(Table table)
    {
        int bestAttr = -1;
        double bestThreshold = 0.0;
        double bestGain = 0.0;

        for (int i = 0; i < initialTable.attributeName.size() - 1; i++)
        {
            if (initialTable.attributeName[i] == "Id")
                continue;

            if (initialTable.isNumericalAttribute[i])
            {
                auto result = getBestThreshold(table, i);
                if (result.second > bestGain)
                {
                    bestGain = result.second;
                    bestAttr = i;
                    bestThreshold = result.first;
                }
            }
            else
            {
                double gain = (this->*currentGainFunction)(table, i);
                if (gain > bestGain)
                {
                    bestGain = gain;
                    bestAttr = i;
                }
            }
        }

        return make_pair(bestAttr, bestThreshold);
    }

    pair<double, double> getBestThreshold(Table table, int attr)
    {
        vector<pair<double, string>> values;

        for (int i = 0; i < table.getDataSize(); i++)
        {
            vector<string> &row = table.getRowData(i);
            double val = stod(row[attr]);
            string label = row.back();
            /*double val = stod(table.data[i][attr]);
            string label = table.data[i].back();*/
            values.push_back(make_pair(val, label));
        }
        // sort all the different values of the numerical datas
        sort(values.begin(), values.end());

        double bestThreshold = 0.0;
        double bestGain = 0.0;

        for (int i = 0; i < values.size() - 1; i++)
        {
            if (values[i].first != values[i + 1].first)
            {
                double threshold = (values[i].first + values[i + 1].first) / 2.0;
                double gain = calcThresholdGain(table, attr, threshold);

                if (gain > bestGain)
                {
                    bestGain = gain;
                    bestThreshold = threshold;
                }
            }
        }

        return make_pair(bestThreshold, bestGain);
    }

    double getIntrinsicValue(Table table, int attributeIndex)
    {
        double ret = 0.0;
        int itemCount = (int)table.getDataSize();

        // Get all the attribute values under a certain attribute index
        map<string, vector<int>> attributeValueMap;
        for (int i = 0; i < table.getDataSize(); i++)
        {
            vector<string> &row = table.getRowData(i);
            attributeValueMap[row[attributeIndex]].push_back(i);
            // attributeValueMap[table.data[i][attributeIndex]].push_back(i);
        }

        // Calculate intrinsic value
        for (auto iter = attributeValueMap.begin(); iter != attributeValueMap.end(); iter++)
        {
            int branchSize = (int)iter->second.size();
            if (branchSize > 0)
            {
                double proportion = (double)branchSize / itemCount;
                ret += -1.0 * proportion * log(proportion) / log(2);
            }
        }
        return ret;
    }
    // IGR
    double getInfoGainRatio(Table table, int attributeIndex)
    {
        double ig = getInfoGain(table, attributeIndex);
        double iv = getIntrinsicValue(table, attributeIndex);

        if (iv == 0.0)
            return 0.0; // Avoid division by zero

        return ig / iv;
    }
    // NWIG
    double getNormalizedWeightedInfoGain(Table table, int attributeIndex)
    {
        double ig = getInfoGain(table, attributeIndex);

        // Calculate number of distinct values (cardinality)
        map<string, int> distinctValues;
        for (int i = 0; i < table.getDataSize(); i++)
        {
            vector<string> &row = table.getRowData(i);
            distinctValues[row[attributeIndex]] = 1;
            // distinctValues[table.data[i][attributeIndex]] = 1;
        }
        int k = distinctValues.size();
        int datasetSize = table.getDataSize();

        // NWIG formula: IG(S,A) / log2(k+1) * (1 - (k-1)/|S|)
        double normalizationFactor = log(k + 1) / log(2);
        double sizePenalty = 1.0 - (double)(k - 1) / datasetSize;

        if (normalizationFactor == 0.0)
            return 0.0;

        return (ig / normalizationFactor) * sizePenalty;
    }

    double calcThresholdGain(Table table, int attr, double threshold)
    {
        Table left, right;

        for (int i = 0; i < table.getDataSize(); i++)
        {
            vector<string> &row = table.getRowData(i);
            double value = stod(row[attr]);
            // double value = stod(table.data[i][attr]);
            if (value <= threshold)
            {
                left.data.push_back(row);
                //left.data.push_back(table.data[i]);
            }
            else
            {
                right.data.push_back(row);
                //right.data.push_back(table.data[i]);
            }
        }

        if (left.getDataSize() == 0 || right.getDataSize() == 0)
            return 0.0;

        /*double totalEntropy = getEntropy(table);
        int totalSize = table.data.size();

        double leftWeight = (double)left.data.size() / totalSize;
        double rightWeight = (double)right.data.size() / totalSize;

        return totalEntropy - (leftWeight * getEntropy(left) + rightWeight * getEntropy(right));*/
        return calcGainForBinarySplit(table, left, right);
    }

    double calcGainForBinarySplit(Table table, Table left, Table right)
    {
        // Determine which gain function to use
        if (currentGainFunction == &DecisionTree::getInfoGain)
        {
            return calcInfoGainForBinarySplit(table, left, right);
        }
        else if (currentGainFunction == &DecisionTree::getInfoGainRatio)
        {
            return calcInfoGainRatioForBinarySplit(table, left, right);
        }
        else if (currentGainFunction == &DecisionTree::getNormalizedWeightedInfoGain)
        {
            return calcNWIGForBinarySplit(table, left, right);
        }
        else
        {
            // Default to Information Gain
            return calcInfoGainForBinarySplit(table, left, right);
        }
    }

    double calcInfoGainForBinarySplit(Table table, Table left, Table right)
    {
        double totalEntropy = getEntropy(table);
        int totalSize = table.getDataSize();

        double leftWeight = (double)left.getDataSize() / totalSize;
        double rightWeight = (double)right.getDataSize() / totalSize;

        return totalEntropy - (leftWeight * getEntropy(left) + rightWeight * getEntropy(right));
    }

    double calcInfoGainRatioForBinarySplit(Table table, Table left, Table right)
    {
        double ig = calcInfoGainForBinarySplit(table, left, right);

        // Calculate Intrinsic Value for binary split
        int totalSize = table.getDataSize();
        double leftProportion = (double)left.getDataSize() / totalSize;
        double rightProportion = (double)right.getDataSize() / totalSize;

        double iv = 0.0;
        if (leftProportion > 0)
            iv += -leftProportion * log(leftProportion) / log(2);
        if (rightProportion > 0)
            iv += -rightProportion * log(rightProportion) / log(2);

        if (iv == 0.0)
            return 0.0; // Avoid division by zero

        return ig / iv;
    }
    double calcNWIGForBinarySplit(Table table, Table left, Table right)
    {
        double ig = calcInfoGainForBinarySplit(table, left, right);

        // For binary split, k = 2 (two branches: left and right)
        int k = 2;
        int datasetSize = table.getDataSize();

        // NWIG formula: IG(S,A) / log2(k+1) * (1 - (k-1)/|S|)
        double normalizationFactor = log(k + 1) / log(2);
        double sizePenalty = 1.0 - (double)(k - 1) / datasetSize;

        if (normalizationFactor == 0.0)
            return 0.0;

        return (ig / normalizationFactor) * sizePenalty;
    }
    // Most likely label
    pair<string, int> getMajorityLabel(Table table)
    {
        string majorLabel = "";
        int majorCount = 0;

        map<string, int> labelCount;
        for (int i = 0; i < table.getDataSize(); i++)
        {
            vector<string> &row = table.getRowData(i);
            labelCount[row.back()]++;
            // labelCount[table.data[i].back()]++;

            if (labelCount[row.back()] > majorCount)
            {
                majorCount = labelCount[row.back()];
                majorLabel = row.back();
                // majorCount = labelCount[table.data[i].back()];
                // majorLabel = table.data[i].back();
            }
        }
        return {majorLabel, majorCount};
    }

    // print the decision tree
    void printTree()
    {
        cout << "Decision Tree Structure:" << endl;
        cout << "========================" << endl;
        printTreeHelper(0, "");
    }

    void printTreeHelper(int nodeIndex, string indent)
    {
        if (tree[nodeIndex].isLeaf)
        {
            cout << indent << "RESULT: " << tree[nodeIndex].label << endl;
            return;
        }

        string attributeName = initialTable.attributeName[tree[nodeIndex].criteriaAttributeIndex];

        if (tree[nodeIndex].isNumerical)
        {
            cout << indent << "IF " << attributeName << " <= " << tree[nodeIndex].threshold << " ?" << endl;
            cout << indent << "  YES:" << endl;
            printTreeHelper(tree[nodeIndex].children[0], indent + "    ");
            cout << indent << "  NO:" << endl;
            printTreeHelper(tree[nodeIndex].children[1], indent + "    ");
        }
        else
        {
            cout << indent << "IF " << attributeName << " = ?" << endl;
            for (int i = 0; i < tree[nodeIndex].children.size(); i++)
            {
                int childIndex = tree[nodeIndex].children[i];
                string attributeValue = tree[childIndex].attributeValue;
                cout << indent << "  WHEN " << attributeName << " = " << attributeValue << ":" << endl;
                printTreeHelper(childIndex, indent + "    ");
            }
        }
    }
};

class InputReader
{
private:
    ifstream fin;
    Table table;

public:
    InputReader(string filename)
    {
        fin.open(filename);
        if (!fin)
        {
            cout << filename << " file could not be opened\n";
            exit(0);
        }
        parse();
    }
    void parse()
    {
        string str;
        // for the first line
        bool isAttributeName = true;
        while (!getline(fin, str).eof())
        {
            vector<string> row;
            stringstream ss(str);
            string cell;
            while (getline(ss, cell, ','))
            {
                // Remove any whitespace
                cell.erase(0, cell.find_first_not_of(" \t\r\n"));
                cell.erase(cell.find_last_not_of(" \t\r\n") + 1);
                row.push_back(cell);
            }

            if (isAttributeName)
            {
                // first column has all the attribute names
                table.attributeName = row;
                isAttributeName = false;
            }
            else
            {
                table.data.push_back(row);
            }
        }
    }

    Table getTable()
    {
        return table;
    }
};
class OutputPrinter
{
private:
    ofstream fout;

public:
    OutputPrinter(string filename)
    {
        fout.open(filename);
        if (!fout)
        {
            cout << filename << " file could not be opened\n";
            exit(0);
        }
    }

    string joinByComma(vector<string> row)
    {
        string ret = "";
        for (int i = 0; i < row.size(); i++)
        {
            ret += row[i];
            if (i != row.size() - 1)
            {
                ret += ',';
            }
        }
        return ret;
    }

    void addLine(string str)
    {
        fout << str << endl;
    }
};

void createTestComparison(DecisionTree &dt, Table &testData)
{
    cout << "\n=== TEST DATA COMPARISON ===" << endl;
    cout << "ID\tActual\t\tGuessed\t\tMatch?" << endl;
    cout << "----------------------------------------" << endl;

    int correct = 0;
    for (int i = 0; i < testData.getDataSize(); i++)
    {
        vector<string> row = testData.getRowData(i);
        string actual = row.back();
        string guessed = dt.guess(row);

        bool match = (actual == guessed);
        if (match)
            correct++;

        cout << row[0] << "\t" << actual << "\t" << guessed << "\t"
             << (match ? "Yes" : "No") << endl;
    }

    cout << "\nAccuracy: " << correct << "/" << testData.getDataSize()
         << " (" << (100.0 * correct / testData.getDataSize()) << "%)" << endl;
}

void test(Table &originalData, string gainMethod = "IG", int maxDepth = 0)
{

    for (int testNum = 1; testNum <= 20; testNum++)
    {
        cout << "\nTest - " << testNum << endl;

        // Split data randomly each time
        pair<Table, Table> splitResult = DataSplitter::splitData(originalData, 0.8);
        Table trainData = splitResult.first;
        Table testData = splitResult.second;
        DecisionTree dt(trainData, gainMethod, maxDepth);
        dt.printTree();
        cout << "Test samples: " << testData.getDataSize() << endl;

        // Create file names with test folder and proper numbering
        string testFolder = ".\\test\\";
        string originalFileName = testFolder + "test_original_" + to_string(testNum) + ".csv";
        string guessedFileName = testFolder + "test_guessed_" + to_string(testNum) + ".csv";
        string comparisonFileName = testFolder + "comparison_" + to_string(testNum) + ".csv";

        // Create output files
        OutputPrinter originalOutput(originalFileName);
        OutputPrinter guessedOutput(guessedFileName);
        OutputPrinter comparisonOutput(comparisonFileName);

        // Write headers
        originalOutput.addLine(originalOutput.joinByComma(testData.attributeName));
        guessedOutput.addLine(originalOutput.joinByComma(testData.attributeName));

        // Create comparison header
        vector<string> comparisonHeader = {"ID", "Actual", "Predicted", "Match"};
        comparisonOutput.addLine(comparisonOutput.joinByComma(comparisonHeader));

        // Write data and calculate accuracy
        int correct = 0;
        for (int i = 0; i < testData.getDataSize(); i++)
        {
            vector<string> row = testData.getRowData(i);
            vector<string> guessedRow = row;

            // Get prediction
            string predicted = dt.guess(row);
            guessedRow.back() = predicted;

            string actual = row.back();
            bool isCorrect = (actual == predicted);
            if (isCorrect)
                correct++;

            // Write to original and guessed files
            originalOutput.addLine(originalOutput.joinByComma(row));
            guessedOutput.addLine(guessedOutput.joinByComma(guessedRow));

            // Write to comparison file
            vector<string> comparisonRow = {
                row[0],
                actual,
                predicted,
                isCorrect ? "Yes" : "No"};
            comparisonOutput.addLine(comparisonOutput.joinByComma(comparisonRow));
        }

        double accuracy = (double)correct / testData.getDataSize() * 100.0;
        cout << "Accuracy: " << correct << "/" << testData.getDataSize()
             << " (" << fixed << setprecision(2) << accuracy << "%)" << endl;
    }
}

int main(int argc, const char *argv[])
{
    string dataInputFile = ".\\CSE 318 Offline 4\\Datasets\\adult.csv";
    string gainMethod = "IG";
    int maxDepth = 0;
    if (argc > 1)
    {
        gainMethod = argv[1];
    }
    if (argc > 2)
    {
        maxDepth = stoi(argv[2]);
    }
    // string resultFile = ".\\Results\\output.txt";
    InputReader dataInputReader(dataInputFile);
    Table originalData = dataInputReader.getTable();
    test(originalData, gainMethod, maxDepth);
    return 0;
}