#ifndef PTXSYNTHESIST_MAIN_WINDOW_H
#define PTXSYNTHESIST_MAIN_WINDOW_H

#include <QMainWindow>
#include <QtCharts/QChartView>
#include <QtCharts/QLineSeries>
#include <QtCharts/QScatterSeries>
#include <QtCharts/QBoxPlotSeries>

#ifdef QT_CHARTS_USE_NAMESPACE
QT_CHARTS_USE_NAMESPACE
#endif

#include <memory>

class QLineEdit;
class CodeEditor;
class QToolBar;
class QTimer;

class SyntaxStyle;
class PTXExecutor;

class ScatterLineSeries
{
public:
    QLineSeries *line_series {};
    QScatterSeries *scatter_series {};

    ScatterLineSeries();

    void set_color(QColor color);
    void add_to_chart(QChart *chart);

    void append(int x, float y);
};

class MainWindow : public QMainWindow
{
  Q_OBJECT

public:
  MainWindow();
  ~MainWindow();

private:
  QLineEdit *options {};
  CodeEditor *cuda {};
  CodeEditor *ptx {};
  QTimer *timer {};
  QToolBar *tool_bar {};

  unsigned int execution_id {};

  ScatterLineSeries min_series;
  ScatterLineSeries median_series;
  ScatterLineSeries max_series;

  QChartView *chart_view {};
  QChart *chart {};

  float min_elapsed = std::numeric_limits<float>::max();
  float max_elapsed = 0.0f;

  SyntaxStyle* syntaxStyle;

  QAction *run_action;
  QAction *interpret_action;

  std::unique_ptr<PTXExecutor> executor;

  void load_style(QString path);

private slots:
  void reset_timer();
  void regen_ptx();
  void interpret();
  void execute();
};

#endif //PTXSYNTHESIST_MAIN_WINDOW_H
