#ifndef PTXSYNTHESIST_SYNTAX_STYLE_H_
#define PTXSYNTHESIST_SYNTAX_STYLE_H_

#include <QObject>
#include <QMap>
#include <QString>
#include <QTextCharFormat>

class SyntaxStyle : public QObject
{
    Q_OBJECT

public:

    /**
     * @brief Constructor.
     * @param parent Pointer to parent QObject
     */
    explicit SyntaxStyle(QObject* parent=nullptr);

    /**
     * @brief Method for loading and parsing
     * style.
     * @param fl Style.
     * @return Success.
     */
    bool load(QString fl);

    /**
     * @brief Method for getting style name.
     * @return Name.
     */
    QString name() const;

    /**
     * @brief Method for checking is syntax style loaded.
     * @return Is loaded.
     */
    bool isLoaded() const;

    /**
     * @brief Method for getting format for property
     * name.
     * @param name Property name.
     * @return Text char format.
     */
    QTextCharFormat getFormat(QString name) const;

    /**
     * @brief Static method for getting default style.
     * @return Pointer to default style.
     */
    static SyntaxStyle* defaultStyle();

private:

    QString m_name;

    QMap<
            QString,
            QTextCharFormat
    > m_data;

    bool m_loaded;
};

#endif