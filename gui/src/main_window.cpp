#include "main_window.h"

#include <QFile>
#include <QTimer>
#include <QLegend>
#include <QToolBar>
#include <QGroupBox>
#include <QLineEdit>
#include <QComboBox>
#include <QTextEdit>
#include <QValueAxis>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QHeaderView>
#include <QDockWidget>
#include <QTableWidget>
#include <QBarCategoryAxis>

#include <QCategoryAxis>

#include "ptx_executor.h"
#include "ptx_generator.h"
#include "ptx_interpreter.h"

#include "code_editor.hpp"
#include "cuda_highlighter.hpp"
#include "ptx_highlighter.hpp"
#include "syntax_style.h"

const char *StyleSheet =
  "QToolBar {"
  " background-color: #282a36;"
  "}"
  "QTextEdit {"
  " background-color: #282A36;"
  "}"
  ""
  "QDockWidget {"
  " background-color: #414450;"
  " color: #414450; "
  "}"
  ""
  "QTabBar::tab {"
  " color: #F8F8F2; "
  " background: #44475a; "
  " border: 0.5px solid #f8f8f2;\n"
  " min-width: 8ex;\n"
  " padding: 2px;\n"
  " margin: 2px 0px 2px 0px;\n"
  "}"
  ""
  "QTabBar::tab:selected, QTabBar::tab:hover {"
  " background: #6272a4; "
  "}"
  ""
  "QTableWidget {"
  " background-color: #414450;"
  " color: #F8F8F2; "
  "}"
  ""
  "QGroupBox {"
  " background-color: #414450;"
  " color: #F8F8F2; "
  "}"
  ""
  "QComboBox {"
  " background-color: #414450;"
  " color: #F8F8F2; "
  "}"
  ""
  "QToolButton {"
  " max-width: 15px; "
  " max-height: 15px; "
  " margin: 8px 4px; "
  "}"
  ""
  "QToolButton:hover {"
  " background-color: #282a36; "
  "}"
  ""
  "QLineEdit {"
  " background-color: #282a36; "
  " color: #F8F8F2; "
  "}"
  ""
  "QScrollBar:vertical {"
  "    padding: 0px 0px 0px 0px;"
  "    background: #282a36;"
  "}"
  "QScrollBar::handle:vertical {"
  "    background: #4c4d56;"
  "}"
  ""
  "QMainWindow {"
  " background-color: #414450; "
  "}";


CUDAPTXPair::CUDAPTXPair(const QString &name, MainWindow *main_window)
  : name(name)
  , timer(new QTimer())
  , cuda(new CodeEditor())
  , ptx(new CodeEditor())
  , main_window(main_window)
{
  cuda->setHighlighter(new CUDAHighlighter());
  ptx->setHighlighter(new PTXHighlighter());

  cuda->setAcceptRichText(false);
  cuda->setPlainText("\n"
                     "// Vector add example\n"
                     "\n"
                     "// threads_in_block = 256\n"
                     "// blocks_in_grid = (n + threads_in_block - 1) // threads_in_block\n"
                     "// iterations = 10\n"
                     "\n"
                     "extern \"C\" __global__ void kernel(\n"
                     "  int n        /* [2 ** x for x in range(18, 27)] */, \n"
                     "  const int *x /* n * 4   */, \n"
                     "  const int *y /* n * 4   */, \n"
                     "  int *result  /* n * 4   */)\n"
                     "{\n"
                     "  for (int i = threadIdx.x + blockIdx.x * blockDim.x;\n"
                     "       i < n;\n"
                     "       i += blockDim.x * gridDim.x) \n"
                     "  {\n"
                     "    result[i] = x[i] + y[i];\n"
                     "  }\n"
                     "}\n");

  QFont jet_brains_mono = QFont("JetBrains Mono", 12);

  options = new QLineEdit();
  options->setText("-lineinfo, --gpu-architecture=compute_75");
  options->setFont(jet_brains_mono);

  QLineEdit *src_name = new QLineEdit();
  src_name->setText(name);
  src_name->setFont(jet_brains_mono);
  src_name->setReadOnly(true);

  QObject::connect(options, &QLineEdit::textChanged, this, &CUDAPTXPair::reset_timer);
  QObject::connect(cuda->document(), &QTextDocument::contentsChanged, this, &CUDAPTXPair::reset_timer);

  cuda->setFont(jet_brains_mono);
  cuda->setAutoIndentation(true);
  cuda->setAutoParentheses(true);
  cuda->setTabReplace(true);
  cuda->setTabReplaceSize(2);

  ptx->setFont(jet_brains_mono);
  ptx->setReadOnly(true);

  QVBoxLayout *v_cuda_layout = new QVBoxLayout();
  v_cuda_layout->addWidget(src_name);
  v_cuda_layout->addWidget(cuda);

  QWidget *cuda_widget = new QWidget();
  cuda_widget->setLayout(v_cuda_layout);

  QVBoxLayout *v_ptx_layout = new QVBoxLayout();
  v_ptx_layout->addWidget(options);
  v_ptx_layout->addWidget(ptx);

  QWidget *ptx_widget = new QWidget();
  ptx_widget->setLayout(v_ptx_layout);

  QDockWidget *cuda_dock_widget = new QDockWidget(name, main_window);
  cuda_dock_widget->setWidget(cuda_widget);
  cuda_dock_widget->setFeatures(cuda_dock_widget->features() & ~QDockWidget::DockWidgetClosable);
  cuda_dock_widget->setAllowedAreas(Qt::AllDockWidgetAreas);
  main_window->addDockWidget(Qt::LeftDockWidgetArea, cuda_dock_widget);

  QDockWidget *ptx_dock_widget = new QDockWidget(name + " PTX", main_window);
  ptx_dock_widget->setWidget(ptx_widget);
  ptx_dock_widget->setFeatures(ptx_dock_widget->features() & ~QDockWidget::DockWidgetClosable);
  ptx_dock_widget->setAllowedAreas(Qt::AllDockWidgetAreas);
  main_window->addDockWidget(Qt::RightDockWidgetArea, ptx_dock_widget);

  timer->setSingleShot(true);
  QObject::connect(timer, &QTimer::timeout, this, &CUDAPTXPair::regen_ptx);
}

std::vector<Measurement> CUDAPTXPair::execute(PTXExecutor *executor)
{
  std::string ptx_code = ptx->toPlainText().toStdString();

  return executor->execute(
    get_params(),
    ptx_code.c_str());
}

void CUDAPTXPair::reset_timer()
{
  timer->start(1000);

  main_window->run_action->setEnabled(false);
  main_window->interpret_action->setEnabled(false);
}

void CUDAPTXPair::regen_ptx()
{
  std::string cuda_source = cuda->toPlainText().toStdString();

  QStringList options_list = options->text().split(',');
  std::vector<std::string> str_options_list(options_list.size());
  std::vector<const char *> c_str_options_list(options_list.size());

  for (int i = 0; i < static_cast<int>(c_str_options_list.size()); i++)
  {
    str_options_list[i] = options_list[i].toStdString();
    c_str_options_list[i] = str_options_list[i].c_str();
  }

  PTXGenerator generator;
  std::optional<PTXCode> ptx_code = generator.gen(cuda_source.c_str(), c_str_options_list);

  if (ptx_code)
  {
    ptx->setPlainText(ptx_code->get_ptx());
  }
  else
  {
    ptx->setPlainText("Compilation error");
  }

  main_window->run_action->setEnabled(true);
  main_window->interpret_action->setEnabled(true);
}

void CUDAPTXPair::load_style(const QString &path, MainWindow *main_window)
{
  QFile fl(path);

  if (!fl.open(QIODevice::ReadOnly))
  {
    return;
  }

  auto style = new SyntaxStyle(main_window);

  if (!style->load(fl.readAll()))
  {
    delete style;
    return;
  }

  cuda->setSyntaxStyle(style);
  ptx->setSyntaxStyle(style);
}

KernelParameter get_int_param(const char *param_name, const QString &cuda_code)
{
  std::string result ("int ");
  result += param_name;
  result += " /* ";

  QString search_pattern = param_name;
  search_pattern += "\\s+=\\s*";

  auto pattern_pos = cuda_code.lastIndexOf(QRegularExpression(search_pattern));
  auto initializer_beg = cuda_code.indexOf('=', pattern_pos) + 1;
  auto initializer_end = cuda_code.indexOf("\n", initializer_beg);

  result += cuda_code.mid(initializer_beg, initializer_end - initializer_beg).toStdString();

  result += " */";

  return KernelParameter(result);
}

std::vector<std::string> split_params(QString params_str)
{
  std::vector<std::string> normalized_params;

  // TODO The whole function is a mess. I'm sorry if you are here.
  //      I'll fix it later
  while(true)
  {
    const bool stop = params_str.back() == ')';

    params_str = params_str.mid(0, params_str.size() - 1);

    if (stop)
    {
      break;
    }
  }

  QStringList lines = params_str.split('\n', Qt::SkipEmptyParts);

  for(QString& line: lines)
  {
    line = line.trimmed();

    if(line.back() == ',')
    {
      line = line.mid(0, line.size() - 1);
    }

    normalized_params.push_back(line.toStdString());
  }

  return normalized_params;
}

std::vector<KernelParameter> CUDAPTXPair::get_params()
{
  QString cuda_code = cuda->toPlainText();

  const int global_idx = cuda_code.lastIndexOf("__global__");
  const int params_start = cuda_code.indexOf('(', global_idx) + 1;
  const int params_len = cuda_code.indexOf('{', params_start) - params_start;

  std::vector<KernelParameter> result;

  for (auto &param: split_params(cuda_code.mid(params_start, params_len)))
  {
    result.emplace_back(param);
  }

  result.push_back(get_int_param("iterations", cuda_code));
  result.push_back(get_int_param("threads_in_block", cuda_code));
  result.push_back(get_int_param("blocks_in_grid", cuda_code));

  return result;
}


ScatterLineSeries::ScatterLineSeries()
    : line_series(new QLineSeries())
    , scatter_series(new QScatterSeries())
{ }

void ScatterLineSeries::set_color(QColor color)
{
    QPen pen = line_series->pen();
    pen.setWidth(3);
    pen.setBrush(QBrush(color));

    line_series->setPen(pen);
    scatter_series->setColor(color);
    // scatter_series->setColor(QColor("#F8F8F2"));
}

void ScatterLineSeries::add_to_chart(QValueAxis *y_axis,
                                     QCategoryAxis *x_axis,
                                     QChart *chart)
{
  chart->addSeries(line_series);
  chart->addSeries(scatter_series);

  line_series->attachAxis(x_axis);
  line_series->attachAxis(y_axis);

  scatter_series->attachAxis(x_axis);
  scatter_series->attachAxis(y_axis);
}

void ScatterLineSeries::append(int x, float y)
{
  line_series->append(static_cast<double>(x), y);
  scatter_series->append(static_cast<double>(x), y);
}

void ScatterLineSeries::set_name(QString name)
{
  line_series->setName(name);
}

MainWindow::MainWindow()
{
  chart = new QChart();
  chart->setBackgroundBrush(QBrush(QColor("#282a36")));

  chart_view = new QChartView(chart);
  chart_view->setBackgroundBrush(QBrush(QColor("#282a36")));
  chart_view->setRenderHint(QPainter::Antialiasing);

  // min_series.set_color("#BD93F9");
  // max_series.set_color("#FFB86C");
  median_series.set_color("#50FA7B");

  y_axis = new QValueAxis();
  x_axis = new QCategoryAxis();
  x_axis->setLabelsPosition(QCategoryAxis::AxisLabelsPositionOnValue);

  chart->addAxis(x_axis, Qt::AlignBottom);
  chart->addAxis(y_axis, Qt::AlignLeft);

  // min_series.add_to_chart(chart);
  // max_series.add_to_chart(chart);
  median_series.add_to_chart(y_axis, x_axis, chart);

  QLegend *legend = chart->legend();
  legend->setLabelColor(QColor("#F8F8F2"));
  legend->setAlignment(Qt::AlignBottom);

  chart->axes(Qt::Vertical).back()->setLabelsColor(QColor("#F8F8F2"));
  chart->axes(Qt::Horizontal).back()->setLabelsColor(QColor("#F8F8F2"));

  tool_bar = addToolBar("Interpreter");
  QWidget* spacer = new QWidget();
  spacer->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

  add_action = new QAction(QIcon(":/icons/plus.png"), "Add", this);
  run_action = new QAction(QIcon(":/icons/play.png"), "Run", this);
  interpret_action = new QAction(QIcon(":/icons/bug.png"), "Interpret", this);

  tool_bar->addAction(add_action);
  tool_bar->addWidget(spacer);
  tool_bar->addAction(run_action);
  tool_bar->addAction(interpret_action);
  tool_bar->setMovable(false);

  QObject::connect(add_action, &QAction::triggered, this, &MainWindow::add);
  QObject::connect(interpret_action, &QAction::triggered, this, &MainWindow::interpret);
  QObject::connect(run_action, &QAction::triggered, this, &MainWindow::execute);

  add();

  setStyleSheet(StyleSheet);
}

MainWindow::~MainWindow() = default;

void MainWindow::interpret()
{
  // TODO
  // PTXInterpreter interpreter;

  // interpreter.interpret(ptx->toPlainText().toStdString());
}

float avg(const std::vector<float> &measurements)
{
    const float sum = std::accumulate(measurements.begin(), measurements.end(), 0.0f);
    return sum / measurements.size();
}

float min(const std::vector<float> &measurements)
{
  return *std::min_element(measurements.begin(), measurements.end());
}

float max(const std::vector<float> &measurements)
{
  return *std::max_element(measurements.begin(), measurements.end());
}

void MainWindow::add()
{
  const std::size_t new_src_id = cuda_ptx_pairs.size();

  auto new_pair = std::make_unique<CUDAPTXPair>(QString("Source #") +
                                                  QString::number(new_src_id),
                                                this);
  new_pair->reset_timer();
  new_pair->load_style(":/style/dracula.xml", this);
  cuda_ptx_pairs.push_back(std::move(new_pair));
}

void MainWindow::execute()
{
  unsigned int last_counter = plot_counter;

  if (!executor)
  {
    executor = std::make_unique<PTXExecutor>();

    QDockWidget *chart_widget = new QDockWidget("chart", this);
    chart_widget->setWidget(chart_view);
    chart_widget->setFeatures(chart_widget->features() & ~QDockWidget::DockWidgetClosable);
    addDockWidget(Qt::BottomDockWidgetArea, chart_widget);
    resizeDocks({chart_widget}, { 2 * height() / 3 }, Qt::Orientation::Vertical);
  }

  std::vector<std::vector<Measurement>> pairs_elapsed_times(cuda_ptx_pairs.size());

  for (std::size_t i = 0; i < cuda_ptx_pairs.size(); i++)
  {
    pairs_elapsed_times[i] = cuda_ptx_pairs[i]->execute(executor.get());
  }

  std::vector<float> elapsed_times;

  if (pairs_elapsed_times.size() == 1)
  {
    const std::vector<Measurement> &measurements = pairs_elapsed_times[0];

    for (std::size_t i = 0; i < measurements.size(); i++)
    {
      elapsed_times.push_back(measurements[i].get_median());
      x_axis->append(QString::number(plot_counter) + ": " + measurements[i].get_name(), plot_counter++);
    }
    x_axis->setRange(-1, plot_counter);
    median_series.set_name("Elapsed time");
  }
  else if (pairs_elapsed_times.size() == 2)
  {
    if (pairs_elapsed_times[0].size() == pairs_elapsed_times[1].size())
    {
      const std::vector<Measurement> &base = pairs_elapsed_times[0];
      const std::vector<Measurement> &cmp = pairs_elapsed_times[1];

      for (std::size_t i = 0; i < base.size(); i++)
      {
        float speedup = base[i].get_median() / cmp[i].get_median();
        elapsed_times.push_back(speedup);

        x_axis->append(QString::number(plot_counter) + ": " + base[i].get_name(), plot_counter++);
      }
    }

    x_axis->setRange(-1, plot_counter);
    median_series.set_name("Speedup: " + cuda_ptx_pairs[0]->name + " / " + cuda_ptx_pairs[1]->name);
  }

  const float min_time = min(elapsed_times);
  const float max_time = max(elapsed_times);

  min_elapsed = std::min(min_elapsed, min_time);
  max_elapsed = std::max(max_elapsed, max_time);

  for (auto &time: elapsed_times)
  {
    median_series.append(last_counter++, time);
  }

  // chart->axes(Qt::Horizontal).back()->setRange(0, execution_id + 1);
  y_axis->setRange(-0.1, max_elapsed + max_elapsed / 4);
}
