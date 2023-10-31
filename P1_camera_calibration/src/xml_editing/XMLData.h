#pragma once
#include "../../precomp.h"

class XMLData
{
public:
	XMLData(const string path, const string filename, bool read_file = false) {
		this->path = path;
		this->filename = filename;
        string filepath = getFilePath();

        if (!read_file) {
            content = "<?xml version = \"1.0\" ?>\n";
            createXMLFile(filepath, content);
        }
        else {
            content = getXMLContent(filepath);
        }
	}

    XMLData(const string path, const string filename, XMLData file_to_copy) {
        this->path = path;
        this->filename = filename;
        string filepath = getFilePath();
        content = getXMLContent(file_to_copy.getFilePath());
        createXMLFile(filepath, content);
    }

    /// <summary>
    /// Adds a tag to the XML file. Optionally adds the tag between enclosing parent tags.
    /// </summary>
    /// <param name="tagName">The name of the tag that will be added.</param>
    /// <param name="parentTagName">The parent tag that will enclose the tag. 
    /// If this value is left empty the tag will be added at the end of the file.</param>
    void AddTag(const string tagName, const string parentTagName = "") {
        if (parentTagName == "") {
            content = std::format("{0}<{1}>\n</{1}>\n", content, tagName);
            return;
        }
        writeValue(parentTagName, std::format("\n\t<{0}>\n\t</{0}>\n", tagName));
    }

    /// <summary>
    /// Returns the full filepath of the XML file.
    /// </summary>
    /// <returns></returns>
    string getFilePath() {
        string filepath = std::format("{0}/{1}", path, filename);

        // Check if the filename ends with .xml.
        if (filepath.substr(filepath.length() - 4, 4) != ".xml")
        {
            filepath.append(".xml");
        }

        return filepath;
    }

    /// <summary>
    /// Finds a tag in an xml file, then returns the content of the tags.
    /// </summary>
    /// <param name="xmlTagName"></param>
    /// <returns></returns>
    string readValue(string xmlTagName) {
        string openingTag = std::format("<{}>", xmlTagName);
        string closingTag = std::format("</{}>", xmlTagName);
        size_t openingPosition = content.find(openingTag);
        size_t closingPosition = content.find(closingTag);
        size_t contentPosition = openingPosition + openingTag.size();
        string value = content.substr(contentPosition, closingPosition - contentPosition);
        return value;
    }

    /// <summary>
    /// Finds a tag in an xml file, then overwrites the content of the tags.
    /// </summary>
    /// <param name="xmlTagName"></param>
    /// <param name="value"></param>
    void writeValue(string xmlTagName, string value) {
        string openingTag = std::format("<{}>", xmlTagName);
        string closingTag = std::format("</{}>", xmlTagName);
        size_t openingPosition = content.find(openingTag);
        size_t closingPosition = content.find(closingTag);
        size_t contentPosition = openingPosition + openingTag.size();
        string start = content.substr(0, contentPosition);
        string end = content.substr(closingPosition, content.size());
        content = std::format("{0}{1}{2}", start, value, end);
    }

    void save() {
        createXMLFile(getFilePath(), content);
    }

    //removeTextFromTag(string xmlTagName, string value)
    //{
    //    string openingTag = std::format("<{}>", xmlTagName);
    //    string closingTag = std::format("</{}>", xmlTagName);
    //    size_t openingPosition = content.find(openingTag);
    //    size_t closingPosition = content.find(closingTag);
    //    size_t contentPosition = openingPosition + openingTag.size();

    //}


private:
    string path;
    string filename;
    string content;

    /// <summary>
    /// Creates an XML file at the given filepath, with the given content.
    /// </summary>
    /// <param name="filepath">The name of the file, and the folder to put it into.</param>
    /// <param name="content">The content of the XML file in string format. Can be empty.</param>
    void createXMLFile(string filepath, const string content = "")
    {
        ofstream XMLFile(filepath);
        XMLFile << content;
        XMLFile.close();
    }

    /// <summary>
    /// Reads XML content from a file and then returns it as a string.
    /// </summary>
    /// <param name="filepath">The filepath of the xml file.</param>
    /// <returns></returns>
    string getXMLContent(string filepath)
    {
        ifstream ifs(filepath);
        string content((istreambuf_iterator<char>(ifs)),
            (istreambuf_iterator<char>()));

        return content;
    }
};

